"""Hermetic tests: everything is generated with h5py in a tmp dir."""

import h5py
import numpy as np
import pytest

from h5coro_hidefix import Index

from conftest import CHUNK, N

DSETS = [
    "/gt1l/heights/h_ph",
    "/gt1l/heights/lat_ph",
    "/gt1l/heights/signal_conf_ph",
    "/meta/plain",
]


@pytest.fixture(scope="module")
def idx(h5file):
    return Index(h5file)


def test_datasets(idx):
    names = idx.datasets()
    assert names == sorted(names)
    for name in DSETS + ["/scalar"]:
        assert name in names


def test_shape_chunkshape_dtype(idx, ref):
    assert idx.shape("/gt1l/heights/h_ph") == (N,)
    assert idx.shape("/gt1l/heights/signal_conf_ph") == (N, 5)
    assert idx.chunk_shape("/gt1l/heights/h_ph") == (CHUNK,)
    assert idx.chunk_shape("/gt1l/heights/signal_conf_ph") == (CHUNK, 5)
    for name in DSETS:
        assert np.dtype(idx.dtype(name)) == ref[name].dtype


def test_read_full(idx, ref):
    for name in DSETS:
        got = idx.read(name)
        expect = ref[name]
        assert got.shape == expect.shape
        assert got.dtype == expect.dtype
        assert got.tobytes() == expect.tobytes()


@pytest.mark.parametrize(
    "start,end",
    [
        (0, 1),
        (0, CHUNK),
        (1, CHUNK + 1),  # crosses a chunk boundary
        (CHUNK - 1, CHUNK + 1),
        (2_500, 7_501),
        (N - 1, N),
        (N - CHUNK - 3, N),  # includes the partial final region
    ],
)
def test_hyperslab(idx, ref, start, end):
    for name in DSETS[:3]:
        got = idx.read(name, start, end)
        expect = ref[name][start:end]
        assert got.shape == expect.shape
        assert got.dtype == expect.dtype
        assert got.tobytes() == expect.tobytes()


def test_no_squeeze(idx, ref):
    """CRITICAL: a length-1 dim-0 slice keeps its full rank (unlike upstream)."""
    got = idx.read("/gt1l/heights/signal_conf_ph", 3, 4)
    assert got.shape == (1, 5)
    assert got.tobytes() == ref["/gt1l/heights/signal_conf_ph"][3:4].tobytes()
    got = idx.read("/gt1l/heights/h_ph", 5, 6)
    assert got.shape == (1,)


def test_empty_slice(idx):
    got = idx.read("/gt1l/heights/signal_conf_ph", 7, 7)
    assert got.shape == (0, 5)
    assert got.dtype == np.int8
    assert idx.read("/gt1l/heights/h_ph", 0, 0).shape == (0,)


def test_default_bounds(idx, ref):
    name = "/gt1l/heights/h_ph"
    assert idx.read(name, 9_000).shape == (1_000,)
    got = idx.read(name, end=10)
    assert got.tobytes() == ref[name][:10].tobytes()


def test_save_load_roundtrip(idx, ref, tmp_path):
    p = tmp_path / "index.bin"
    idx.save(p)
    assert p.stat().st_size > 0
    loaded = Index.load(p)
    assert loaded.datasets() == idx.datasets()
    assert loaded.source == idx.source
    for name in DSETS:
        a1, s1, o1 = idx.chunks(name)
        a2, s2, o2 = loaded.chunks(name)
        np.testing.assert_array_equal(a1, a2)
        np.testing.assert_array_equal(s1, s2)
        np.testing.assert_array_equal(o1, o2)
        got = loaded.read(name)
        assert got.tobytes() == ref[name].tobytes()


def test_load_source_override(idx, h5file, ref, tmp_path):
    p = tmp_path / "index.bin"
    idx.save(p)
    other = tmp_path / "renamed.h5"
    other.write_bytes(h5file.read_bytes())
    loaded = Index.load(p, source=other)
    assert str(loaded.source) == str(other)
    got = loaded.read("/gt1l/heights/h_ph", 100, 2_100)
    assert got.tobytes() == ref["/gt1l/heights/h_ph"][100:2_100].tobytes()


def test_chunks_match_h5py(idx, h5file):
    """Chunk table must agree with h5py's B-tree walk, chunk for chunk."""
    with h5py.File(h5file, "r") as f:
        for name in DSETS[:3]:
            dsid = f[name].id
            nchunks = dsid.get_num_chunks()
            expect = {}
            for i in range(nchunks):
                info = dsid.get_chunk_info(i)
                expect[tuple(info.chunk_offset)] = (info.byte_offset, info.size)
            addrs, sizes, offsets = idx.chunks(name)
            assert addrs.dtype == sizes.dtype == offsets.dtype == np.uint64
            assert len(addrs) == nchunks
            assert offsets.shape == (nchunks, len(f[name].shape))
            for a, s, o in zip(addrs, sizes, offsets):
                assert expect[tuple(o)] == (a, s)


def test_chunks_contiguous(idx, h5file):
    """A contiguous dataset is one pseudo-chunk covering the whole extent."""
    addrs, sizes, offsets = idx.chunks("/meta/plain")
    assert len(addrs) == 1
    assert sizes[0] == 37 * 4
    assert offsets.shape == (1, 1) and offsets[0, 0] == 0
    with h5py.File(h5file, "r") as f:
        assert addrs[0] == f["/meta/plain"].id.get_offset()


def test_errors(idx):
    with pytest.raises(KeyError):
        idx.read("/nope")
    with pytest.raises(KeyError):
        idx.chunks("/also/nope")
    with pytest.raises(ValueError):
        idx.read("/gt1l/heights/h_ph", 5, 4)
    with pytest.raises(ValueError):
        idx.read("/gt1l/heights/h_ph", 0, N + 1)
    with pytest.raises(ValueError):
        idx.read("/scalar")


def test_load_garbage_raises(tmp_path):
    p = tmp_path / "garbage.bin"
    p.write_bytes(b"not an index")
    with pytest.raises(Exception):
        Index.load(p)


def _fetch(h5file, addrs, sizes):
    """Simulate the object-store fetch: ranged reads from the local file."""
    out = []
    with open(h5file, "rb") as f:
        for addr, size in zip(addrs, sizes):
            f.seek(int(addr))
            out.append(f.read(int(size)))
    return out


@pytest.mark.parametrize(
    "start,end",
    [
        (None, None),  # full dataset
        (0, 1),
        (1, CHUNK + 1),  # crosses a chunk boundary
        (CHUNK - 1, CHUNK + 1),
        (2_500, 7_501),
        (N - CHUNK - 3, N),  # includes the partial final region
        (7, 7),  # empty range -> zero chunks, zero buffers
    ],
)
def test_read_from_buffers_matches_read(idx, ref, h5file, start, end):
    for name in DSETS[:3]:  # 1-D f8 and 2-D (n, 5) i1
        addrs, sizes, offsets = idx.read_plan(name, start, end)
        assert len(addrs) == len(sizes) == len(offsets)
        buffers = _fetch(h5file, addrs, sizes)
        got = idx.read_from_buffers(name, buffers, start, end)
        via_read = idx.read(name, start, end)
        expect = ref[name][start:end]
        assert got.shape == via_read.shape == expect.shape
        assert got.dtype == via_read.dtype == expect.dtype
        assert got.tobytes() == via_read.tobytes() == expect.tobytes()


def test_read_from_buffers_contiguous(idx, ref, h5file):
    """The contiguous (single pseudo-chunk, unfiltered) dataset round-trips."""
    name = "/meta/plain"
    for start, end in [(None, None), (0, 1), (5, 20), (36, 37), (3, 3)]:
        addrs, sizes, _ = idx.read_plan(name, start, end)
        buffers = _fetch(h5file, addrs, sizes)
        got = idx.read_from_buffers(name, buffers, start, end)
        assert got.tobytes() == ref[name][start:end].tobytes()


def test_read_plan_subsets_chunks(idx):
    name = "/gt1l/heights/signal_conf_ph"
    all_addrs, all_sizes, all_offsets = idx.chunks(name)
    addrs, sizes, offsets = idx.read_plan(name, CHUNK, 3 * CHUNK)
    assert len(addrs) == 2  # rows [CHUNK, 3*CHUNK) -> exactly chunks 1 and 2
    np.testing.assert_array_equal(addrs, all_addrs[1:3])
    np.testing.assert_array_equal(sizes, all_sizes[1:3])
    np.testing.assert_array_equal(offsets, all_offsets[1:3])
    # empty range plans no chunks
    addrs, sizes, offsets = idx.read_plan(name, 5, 5)
    assert len(addrs) == 0 and offsets.shape == (0, 2)


def test_read_from_buffers_accepts_bytes_like(idx, ref, h5file):
    name = "/gt1l/heights/h_ph"
    addrs, sizes, _ = idx.read_plan(name, 0, CHUNK)
    (buf,) = _fetch(h5file, addrs, sizes)
    expect = ref[name][:CHUNK].tobytes()
    for wrap in (bytes, bytearray, memoryview):
        got = idx.read_from_buffers(name, [wrap(buf)], 0, CHUNK)
        assert got.tobytes() == expect


def test_read_from_buffers_wrong_count(idx, h5file):
    name = "/gt1l/heights/h_ph"
    addrs, sizes, _ = idx.read_plan(name, 0, 2 * CHUNK)
    buffers = _fetch(h5file, addrs, sizes)
    with pytest.raises(ValueError, match="expected 2 buffers"):
        idx.read_from_buffers(name, buffers[:1], 0, 2 * CHUNK)
    with pytest.raises(ValueError, match="expected 2 buffers"):
        idx.read_from_buffers(name, buffers + buffers[:1], 0, 2 * CHUNK)


def test_read_from_buffers_wrong_size(idx, h5file):
    name = "/gt1l/heights/h_ph"
    addrs, sizes, _ = idx.read_plan(name, 0, CHUNK)
    (buf,) = _fetch(h5file, addrs, sizes)
    with pytest.raises(ValueError, match="stores"):
        idx.read_from_buffers(name, [buf[:-1]], 0, CHUNK)
    with pytest.raises(ValueError, match="not bytes-like"):
        idx.read_from_buffers(name, [12345], 0, CHUNK)


def test_read_from_buffers_corrupted(idx, h5file):
    name = "/gt1l/heights/h_ph"
    addrs, sizes, _ = idx.read_plan(name, 0, CHUNK)
    (buf,) = _fetch(h5file, addrs, sizes)
    corrupted = bytes(len(buf))  # right length, garbage (invalid deflate)
    with pytest.raises(Exception):
        idx.read_from_buffers(name, [corrupted], 0, CHUNK)


def test_read_with_missing_source_raises(idx, tmp_path):
    p = tmp_path / "index.bin"
    idx.save(p)
    loaded = Index.load(p, source=tmp_path / "missing.h5")
    with pytest.raises(Exception):
        loaded.read("/gt1l/heights/h_ph", 0, 1)

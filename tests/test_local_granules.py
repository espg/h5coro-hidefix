"""Local-only tests against real ATL03 granules (skipped when absent).

Spot-checks a real granule byte-for-byte against h5py. No granule data is
committed to this repository.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from h5coro_hidefix import Index

GRANULE_DIR = Path(
    os.environ.get(
        "H5CORO_HIDEFIX_GRANULE_DIR",
        Path.home() / "ignore" / "zagg_neon_atl03_test_shard" / "granules",
    )
)

pytestmark = pytest.mark.skipif(
    not GRANULE_DIR.is_dir() or not sorted(GRANULE_DIR.glob("*.h5")),
    reason=f"no local granules under {GRANULE_DIR}",
)

BEAM_DSETS = [
    "/gt1l/heights/lat_ph",
    "/gt1l/heights/lon_ph",
    "/gt1l/heights/h_ph",
    "/gt1l/heights/signal_conf_ph",
]


@pytest.fixture(scope="module")
def granule():
    return sorted(GRANULE_DIR.glob("*.h5"))[0]


@pytest.fixture(scope="module")
def idx(granule):
    return Index(granule)


def _present(idx):
    names = set(idx.datasets())
    dsets = [d for d in BEAM_DSETS if d in names]
    assert dsets, "granule has no gt1l heights datasets"
    return dsets


def test_reads_match_h5py(idx, granule):
    h5py = pytest.importorskip("h5py")
    with h5py.File(granule, "r") as f:
        for name in _present(idx):
            expect = f[name][...]
            got = idx.read(name)
            assert got.shape == expect.shape
            assert got.dtype == expect.dtype
            assert got.tobytes() == expect.tobytes()
            # hyperslabs, including a no-squeeze length-1 row range
            n = expect.shape[0]
            for start, end in [(0, 1), (n // 3, n // 3 + 10_000), (n - 7, n)]:
                got = idx.read(name, start, end)
                ref = f[name][start:end]
                assert got.shape == ref.shape
                assert got.tobytes() == ref.tobytes()


def test_save_load_roundtrip(idx, granule, tmp_path):
    p = tmp_path / "granule.idx"
    idx.save(p)
    loaded = Index.load(p, source=granule)
    name = _present(idx)[0]
    a, b = idx.read(name, 1_000, 51_000), loaded.read(name, 1_000, 51_000)
    assert a.tobytes() == b.tobytes()


def test_chunks_match_h5py(idx, granule):
    h5py = pytest.importorskip("h5py")
    name = _present(idx)[0]
    addrs, sizes, offsets = idx.chunks(name)
    with h5py.File(granule, "r") as f:
        dsid = f[name].id
        nchunks = dsid.get_num_chunks()
        assert len(addrs) == nchunks
        for i in (0, nchunks // 2, nchunks - 1):
            info = dsid.get_chunk_info(i)
            j = np.nonzero(addrs == info.byte_offset)[0]
            assert len(j) == 1
            j = j[0]
            assert sizes[j] == info.size
            assert tuple(offsets[j]) == tuple(info.chunk_offset)

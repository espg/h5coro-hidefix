"""Hermetic tests for the zagg sidecar backend pieces (no zagg installed).

Protocol-shape conformance runs against ``zagg_protocol_stub`` (pinned to
englacial/zagg 87b941ed29618ba9b1dfee1ec2668392cd1f9ac3 — see that file);
manifest parsing + reconstruction round-trips run against a synthetic
parquet manifest in zagg's write-back schema.
"""

import inspect
import json
import subprocess
import sys
import types

import numpy as np
import pytest

from h5coro_hidefix import Index
from h5coro_hidefix.manifest import MANIFEST_COLUMNS, datasets_from_manifest

DSETS = ["/gt1l/heights/h_ph", "/gt1l/heights/signal_conf_ph", "/meta/plain"]


def test_base_import_pulls_no_heavy_deps():
    """`import h5coro_hidefix` alone must not import zagg/pandas/pyarrow."""
    code = (
        "import sys; import h5coro_hidefix; "
        "bad = [m for m in ('zagg', 'pandas', 'pyarrow', 'h5coro_hidefix.zagg_backend') "
        "if m in sys.modules]; "
        "assert not bad, bad; print('clean')"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "clean"


# ---------------------------------------------------------------------------
# Manifest construction helpers (zagg's write-back schema, gzip as BOOL)
# ---------------------------------------------------------------------------


def _manifest_columns(idx, names):
    """Build manifest columns exactly as zagg's inline write-back emits them:
    dtype in np.dtype().str form, JSON tuple cells, gzip as a boolean."""
    cols = {c: [] for c in MANIFEST_COLUMNS}
    for name in names:
        addrs, sizes, offsets = idx.chunks(name)
        filt = idx.filters(name)
        shape = list(idx.shape(name))
        chunk_shape = list(idx.chunk_shape(name))
        grid = [-(-d // c) for d, c in zip(shape, chunk_shape)]
        step = [1] * len(grid)
        for d in range(len(grid) - 2, -1, -1):
            step[d] = grid[d + 1] * step[d + 1]
        for a, s, off in zip(addrs, sizes, offsets):
            cols["dataset"].append(name)
            cols["chunk_idx"].append(
                sum((int(o) // c) * st for o, c, st in zip(off, chunk_shape, step))
            )
            cols["elem_start"].append(int(off[0]))
            cols["elem_end"].append(min(int(off[0]) + chunk_shape[0], shape[0]))
            cols["byte_offset"].append(int(a))
            cols["nbytes"].append(int(s))
            cols["filter_mask"].append(0)
            cols["chunk_offset"].append(json.dumps([int(o) for o in off]))
            cols["dtype"].append(np.dtype(idx.dtype(name)).str)
            cols["shape"].append(json.dumps(shape))
            cols["chunk_shape"].append(json.dumps(chunk_shape))
            cols["gzip"].append(filt["gzip"] is not None)  # BOOL, per the contract
            cols["shuffle"].append(filt["shuffle"])
    return cols


@pytest.fixture(scope="module")
def idx(h5file):
    return Index(h5file)


def _assert_reads_identical(vidx, idx, h5file, names):
    for name in names:
        n = idx.shape(name)[0]
        spans = [(None, None), (0, 1), (max(0, n - 3), n), (min(3, n), min(3, n))]
        for start, end in spans:
            expect = idx.read(name, start, end)
            got = vidx.read(name, start, end)
            assert got.shape == expect.shape
            assert got.dtype == expect.dtype
            assert got.tobytes() == expect.tobytes()
            addrs, sizes, _ = vidx.read_plan(name, start, end)
            with open(h5file, "rb") as f:
                bufs = []
                for a, s in zip(addrs, sizes):
                    f.seek(int(a))
                    bufs.append(f.read(int(s)))
            got = vidx.read_from_buffers(name, bufs, start, end)
            assert got.tobytes() == expect.tobytes()


def test_manifest_columns_round_trip(idx, h5file):
    cols = _manifest_columns(idx, DSETS)
    specs = datasets_from_manifest(cols)
    assert sorted(specs) == sorted(DSETS)
    vidx = Index.from_chunks(h5file, specs)
    _assert_reads_identical(vidx, idx, h5file, DSETS)


def test_manifest_parquet_round_trip(idx, h5file, tmp_path):
    """The real interchange: columns -> parquet (pyarrow) -> columns -> Index."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    cols = _manifest_columns(idx, DSETS)
    path = tmp_path / "granule.parquet"
    pq.write_table(pa.table(cols), path)
    back = pq.read_table(path).to_pydict()
    assert back["gzip"][0] in (True, False)  # bool column survives parquet
    vidx = Index.from_chunks(h5file, datasets_from_manifest(back))
    _assert_reads_identical(vidx, idx, h5file, DSETS)


def test_manifest_skips_unmappable_dtype(idx):
    cols = _manifest_columns(idx, DSETS)
    cols["dtype"] = ["" for _ in cols["dtype"]]
    assert datasets_from_manifest(cols) == {}


def test_manifest_missing_column(idx):
    cols = _manifest_columns(idx, DSETS[:1])
    del cols["byte_offset"]
    with pytest.raises(KeyError, match="byte_offset"):
        datasets_from_manifest(cols)


def test_manifest_malformed_json(idx):
    cols = _manifest_columns(idx, DSETS[:1])
    cols["chunk_offset"][0] = "not json"
    with pytest.raises(ValueError, match="malformed JSON"):
        datasets_from_manifest(cols)


# ---------------------------------------------------------------------------
# Protocol conformance against the pinned stub (no zagg env needed)
# ---------------------------------------------------------------------------


@pytest.fixture()
def sidecar_cls(monkeypatch):
    """Import h5coro_hidefix.zagg_backend against the pinned protocol stub."""
    import zagg_protocol_stub as stub

    zagg_pkg = types.ModuleType("zagg")
    zagg_index = types.ModuleType("zagg.index")
    zagg_index.VirtualIndex = stub.VirtualIndex
    zagg_pkg.index = zagg_index
    monkeypatch.setitem(sys.modules, "zagg", zagg_pkg)
    monkeypatch.setitem(sys.modules, "zagg.index", zagg_index)
    monkeypatch.delitem(sys.modules, "h5coro_hidefix.zagg_backend", raising=False)
    from h5coro_hidefix.zagg_backend import SidecarIndex

    yield SidecarIndex
    monkeypatch.delitem(sys.modules, "h5coro_hidefix.zagg_backend", raising=False)


class TestProtocolShape:
    def test_subclass_and_registry_attrs(self, sidecar_cls):
        import zagg_protocol_stub as stub

        assert issubclass(sidecar_cls, stub.VirtualIndex)
        assert sidecar_cls.name == "sidecar"
        assert sidecar_cls.config_keys == frozenset({"store", "on_miss"})
        assert sidecar_cls.required_config_keys == frozenset({"store"})
        assert sidecar_cls.required_config_keys <= sidecar_cls.config_keys

    def test_method_signatures_match_protocol(self, sidecar_cls):
        import zagg_protocol_stub as stub

        for meth in ("read_group", "finish_granule"):
            got = inspect.signature(getattr(sidecar_cls, meth))
            want = inspect.signature(getattr(stub.VirtualIndex, meth))
            assert list(got.parameters) == list(want.parameters), meth
            for p in want.parameters.values():
                assert got.parameters[p.name].default == p.default, (meth, p.name)

    def test_validate_config(self, sidecar_cls):
        ds_ok = {"read_plan": {"spatial_index": "segments"}}
        sidecar_cls.validate_index_config({"backend": "sidecar", "store": "/x"}, ds_ok)
        with pytest.raises(ValueError, match="requires 'store'"):
            sidecar_cls.validate_index_config({"backend": "sidecar"}, ds_ok)
        with pytest.raises(ValueError, match="on_miss"):
            sidecar_cls.validate_index_config(
                {"backend": "sidecar", "store": "/x", "on_miss": "explode"}, ds_ok
            )
        with pytest.raises(ValueError, match="spatial_index"):
            sidecar_cls.validate_index_config(
                {"backend": "sidecar", "store": "/x"}, {"read_plan": {}}
            )
        with pytest.raises(ValueError, match="mutually exclusive"):
            sidecar_cls.validate_index_config(
                {"backend": "sidecar", "store": "/x"},
                {"read_plan": {"spatial_index": "s", "chunk_boundaries": "p"}},
            )

    def test_from_index_config(self, sidecar_cls):
        b = sidecar_cls.from_index_config({"backend": "sidecar", "store": "/s"})
        assert (b.store, b.on_miss) == ("/s", "fallback")
        b = sidecar_cls.from_index_config(
            {"backend": "sidecar", "store": "/s", "on_miss": "build"}
        )
        assert b.on_miss == "build"

    def test_read_fn_serves_planned_slices(self, sidecar_cls, idx, h5file):
        """The addressing seam end-to-end with a fake credentialed driver:
        read_fn(path, hyperslice) must equal direct Index reads."""

        class _FakeH5:
            resource = str(h5file)

            @staticmethod
            def ioRequest(pos, size, caching=True, prefetch=False):  # noqa: N802
                with open(h5file, "rb") as f:
                    f.seek(pos)
                    return f.read(size)

        backend = sidecar_cls(store="unused")
        cols = _manifest_columns(idx, DSETS)
        vidx = Index.from_chunks(h5file, datasets_from_manifest(cols))
        read_fn = backend._read_fn_for(vidx, _FakeH5)
        n = idx.shape("/gt1l/heights/h_ph")[0]
        for path in DSETS[:2]:
            full = read_fn(path)
            assert full.tobytes() == idx.read(path).tobytes()
            got = read_fn(path, [(5, 800), (2500, 2501)])
            expect = np.concatenate([idx.read(path, 5, 800), idx.read(path, 2500, 2501)])
            assert got.tobytes() == expect.tobytes()
        one = read_fn("/gt1l/heights/h_ph", [(n - 1, n)])
        assert one.shape == (1,)

    def test_pooled_fetches_byte_identical_and_ordered(self, sidecar_cls, idx, h5file):
        """workers=8 must equal workers=1 byte-for-byte even when fetch
        completion order is scrambled -- read_from_buffers depends on the
        buffers arriving in read_plan order (zagg issue #170)."""
        import threading
        import time

        class _JitteredH5:
            resource = str(h5file)
            calls: list = []
            lock = threading.Lock()

            @classmethod
            def ioRequest(cls, pos, size, caching=True, prefetch=False):  # noqa: N802
                # Later offsets return FIRST: completion order is the exact
                # reverse of submission order, so any ordering bug shows.
                with cls.lock:
                    cls.calls.append(pos)
                    rank = len(cls.calls)
                time.sleep(max(0.0, 0.03 - 0.005 * rank))
                with open(h5file, "rb") as f:
                    f.seek(pos)
                    return f.read(size)

        backend = sidecar_cls(store="unused")
        cols = _manifest_columns(idx, DSETS)
        vidx = Index.from_chunks(h5file, datasets_from_manifest(cols))
        serial = backend._read_fn_for(vidx, _JitteredH5, workers=1)
        pooled = backend._read_fn_for(vidx, _JitteredH5, workers=8)
        for path in DSETS[:2]:
            assert pooled(path).tobytes() == serial(path).tobytes()
            hs = [(5, 800), (2500, 2501)]
            assert pooled(path, hs).tobytes() == serial(path, hs).tobytes()
        assert len(_JitteredH5.calls) > 0

    def test_none_buffer_surfaces_as_os_error(self, sidecar_cls, idx, h5file):
        """h5coro drivers swallow exceptions and return None on failed ranged
        reads; that must raise OSError, not fail inside the decoder with a
        buffer-type error (zagg PR #173 review lesson)."""
        class _FlakyH5:
            resource = str(h5file)

            @staticmethod
            def ioRequest(pos, size, caching=True, prefetch=False):  # noqa: N802
                return None

        backend = sidecar_cls(store="unused")
        cols = _manifest_columns(idx, DSETS)
        vidx = Index.from_chunks(h5file, datasets_from_manifest(cols))
        read_fn = backend._read_fn_for(vidx, _FlakyH5, workers=4)
        with pytest.raises(OSError, match="ranged read failed"):
            read_fn(DSETS[0], [(0, 10)])

    def test_fetch_workers_validation(self):
        from h5coro_hidefix.zagg_backend import _fetch_workers

        assert _fetch_workers({}) == 8
        assert _fetch_workers(None) == 8
        assert _fetch_workers({"read_workers": 3}) == 3
        for bad in (0, -1, True, "eight", 2.5):
            with pytest.raises(ValueError, match="read_workers"):
                _fetch_workers({"read_workers": bad})

    def test_read_fn_missing_dataset_raises_miss(self, sidecar_cls, idx, h5file):
        from h5coro_hidefix.zagg_backend import _ManifestMiss

        backend = sidecar_cls(store="unused")
        cols = _manifest_columns(idx, DSETS[:1])
        vidx = Index.from_chunks(h5file, datasets_from_manifest(cols))
        read_fn = backend._read_fn_for(vidx, None)
        with pytest.raises(_ManifestMiss):
            read_fn("/not/in/manifest", [(0, 1)])

    def test_granule_id_derivation(self, sidecar_cls):
        from h5coro_hidefix.zagg_backend import _granule_id

        gid = "ATL03_20190105163308_01260202_007_01"
        assert _granule_id(f"s3://bucket/prefix/{gid}.h5") == gid
        assert _granule_id(f"https://host/path/{gid}.h5?A-userid=x") == gid
        assert _granule_id(f"/local/dir/{gid}.h5") == gid

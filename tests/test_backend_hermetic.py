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
        # Flat (read-plan-less) sources are accepted -- both read routes are
        # served since the zagg 0.15 full-read seam (mirrors inline).
        sidecar_cls.validate_index_config({"backend": "sidecar", "store": "/x"}, {})
        sidecar_cls.validate_index_config(
            {"backend": "sidecar", "store": "/x"}, {"read_plan": {}}
        )
        with pytest.raises(ValueError, match="mutually exclusive"):
            sidecar_cls.validate_index_config(
                {"backend": "sidecar", "store": "/x"},
                {"read_plan": {"spatial_index": "s", "chunk_boundaries": "p"}},
            )
        # read_workers is gated at backend resolution (submission time), not
        # left to blow up inside a per-group read.
        for bad in (0, -1, True, "eight", 2.5):
            with pytest.raises(ValueError, match="read_workers"):
                sidecar_cls.validate_index_config(
                    {"backend": "sidecar", "store": "/x"}, {"read_workers": bad}
                )
        sidecar_cls.validate_index_config(
            {"backend": "sidecar", "store": "/x"}, {"read_workers": 4}
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
            # (500, 3500) spans multiple chunks so the hyperslab arm also
            # exercises the pool, not just the single-chunk serial branch.
            hs = [(5, 800), (500, 3500), (2500, 2501)]
            assert pooled(path, hs).tobytes() == serial(path, hs).tobytes()
        assert len(_JitteredH5.calls) > 0

    def test_fetch_budget_caps_concurrent_gets(
        self, sidecar_cls, idx, h5file, monkeypatch
    ):
        """The process-global fetch semaphore caps in-flight ioRequest calls
        at the budget no matter how wide the per-read pool is (issue #5 item
        2): workers=32 over many chunks, budget shrunk to 4 -- peak observed
        concurrency must never exceed 4, and bytes stay identical."""
        import threading
        import time

        zb = sys.modules["h5coro_hidefix.zagg_backend"]
        monkeypatch.setattr(zb, "_FETCH_SEMAPHORE", threading.BoundedSemaphore(4))

        class _TrackingH5:
            resource = str(h5file)
            lock = threading.Lock()
            active = 0
            peak = 0

            @classmethod
            def ioRequest(cls, pos, size, caching=True, prefetch=False):  # noqa: N802
                with cls.lock:
                    cls.active += 1
                    cls.peak = max(cls.peak, cls.active)
                time.sleep(0.005)
                try:
                    with open(h5file, "rb") as f:
                        f.seek(pos)
                        return f.read(size)
                finally:
                    with cls.lock:
                        cls.active -= 1

        backend = sidecar_cls(store="unused")
        cols = _manifest_columns(idx, DSETS)
        vidx = Index.from_chunks(h5file, datasets_from_manifest(cols))
        path = DSETS[0]
        addrs, _, _ = vidx.read_plan(path, None, None)
        assert len(addrs) > 4  # enough chunks that an uncapped pool would exceed 4
        read_fn = backend._read_fn_for(vidx, _TrackingH5, workers=32)
        assert read_fn(path).tobytes() == idx.read(path).tobytes()
        assert _TrackingH5.peak <= 4
        assert _TrackingH5.peak >= 2  # the pool did run fetches concurrently

    def test_fetch_budget_contention_across_read_fns(
        self, sidecar_cls, idx, h5file, monkeypatch
    ):
        """Two read_fns contending for a budget of 2 (two threads, different
        datasets) both complete byte-identical -- the semaphore is only ever
        held around the GET itself, so contention cannot deadlock."""
        import threading

        zb = sys.modules["h5coro_hidefix.zagg_backend"]
        monkeypatch.setattr(zb, "_FETCH_SEMAPHORE", threading.BoundedSemaphore(2))

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
        results, errors = {}, []

        def read(path):
            try:
                results[path] = backend._read_fn_for(vidx, _FakeH5, workers=8)(path)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=read, args=(p,)) for p in DSETS[:2]]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        assert not any(t.is_alive() for t in threads)  # no deadlock
        assert not errors
        for path in DSETS[:2]:
            assert results[path].tobytes() == idx.read(path).tobytes()

    def test_fetch_budget_env_override(self, sidecar_cls, monkeypatch):
        """ZAGG_HIDEFIX_FETCH_BUDGET is parsed defensively: positive ints
        win, anything else falls back to the module default."""
        zb = sys.modules["h5coro_hidefix.zagg_backend"]
        monkeypatch.delenv("ZAGG_HIDEFIX_FETCH_BUDGET", raising=False)
        assert zb._fetch_budget() == zb._FETCH_BUDGET
        monkeypatch.setenv("ZAGG_HIDEFIX_FETCH_BUDGET", "128")
        assert zb._fetch_budget() == 128
        for bad in ("0", "-3", "eight", "2.5", ""):
            monkeypatch.setenv("ZAGG_HIDEFIX_FETCH_BUDGET", bad)
            assert zb._fetch_budget() == zb._FETCH_BUDGET

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
        # Full read: many covering chunks, so the POOLED branch's mid-stream
        # re-raise (pool.map) is what gets exercised, not the n <= 1 serial
        # short-circuit (review finding on this PR).
        with pytest.raises(OSError, match="ranged read failed"):
            read_fn(DSETS[0])
        # Single-chunk span: the serial branch raises identically.
        with pytest.raises(OSError, match="ranged read failed"):
            read_fn(DSETS[0], [(0, 10)])

    def _stub_read_module(self, monkeypatch, with_seam=True):
        """Install a fake ``zagg.processing.read`` capturing the route taken.

        ``with_seam=True`` mirrors zagg >= 0.15 (``_read_group_full`` accepts
        ``read_fn``); ``False`` mirrors an older zagg (no seam parameter).
        """
        import sys
        import types

        calls = {}
        read_mod = types.ModuleType("zagg.processing.read")

        def _planned_read_group(h5obj, group, ds, shard_key, grid, arrow=False, read_fn=None):
            calls["route"] = "planned"
            calls["read_fn"] = read_fn
            return "planned-sentinel"

        def _validate_planned_config(ds):
            pass

        if with_seam:

            def _read_group_full(h5obj, group, ds, shard_key, grid, arrow=False, read_fn=None):
                calls["route"] = "full"
                calls["read_fn"] = read_fn
                return "full-sentinel"
        else:

            def _read_group_full(h5obj, group, ds, shard_key, grid, arrow=False):
                calls["route"] = "full-legacy"
                return "full-legacy-sentinel"

        read_mod._planned_read_group = _planned_read_group
        read_mod._validate_planned_config = _validate_planned_config
        read_mod._read_group_full = _read_group_full
        proc_mod = types.ModuleType("zagg.processing")
        proc_mod.read = read_mod
        monkeypatch.setitem(sys.modules, "zagg.processing", proc_mod)
        monkeypatch.setitem(sys.modules, "zagg.processing.read", read_mod)
        return calls

    def test_flat_source_routes_to_full_read_seam(
        self, sidecar_cls, idx, h5file, monkeypatch, tmp_path
    ):
        """A read-plan-less (flat) data source takes the compiled full-read
        route (zagg >= 0.15 seam), with a working read_fn attached."""
        from pathlib import Path

        calls = self._stub_read_module(monkeypatch, with_seam=True)

        class _FakeH5:
            resource = str(h5file)

            @staticmethod
            def ioRequest(pos, size, caching=True, prefetch=False):  # noqa: N802
                with open(h5file, "rb") as f:
                    f.seek(pos)
                    return f.read(size)

        backend = sidecar_cls(store=str(tmp_path))
        # Pre-seed the per-granule cache: manifest fetch (obstore) is covered
        # by the parquet round-trip tests; this test pins the ROUTING.
        cols = _manifest_columns(idx, DSETS)
        vidx = Index.from_chunks(str(h5file), datasets_from_manifest(cols))
        backend._cache[Path(str(h5file)).stem] = vidx
        out = backend.read_group(_FakeH5, "gt1l", {}, 1, grid=None)
        assert out == "full-sentinel"
        assert calls["route"] == "full"
        got = calls["read_fn"]("/gt1l/heights/h_ph", [(5, 800)])
        assert got.tobytes() == idx.read("/gt1l/heights/h_ph", 5, 800).tobytes()

    def test_flat_source_on_old_zagg_is_actionable(
        self, sidecar_cls, idx, h5file, monkeypatch, tmp_path
    ):
        """Against a pre-seam zagg (no read_fn on _read_group_full), a flat
        source fails loudly with the upgrade path, never silently degrades."""
        self._stub_read_module(monkeypatch, with_seam=False)
        backend = sidecar_cls(store=str(tmp_path))
        with pytest.raises(ValueError, match="requires zagg >= 0.15"):
            backend.read_group(object(), "gt1l", {}, 1, grid=None)

    def test_build_delegate_constructed_once_under_concurrent_misses(
        self, sidecar_cls, monkeypatch, tmp_path
    ):
        """on_miss: build under concurrent granule reads (zagg PR #183's
        threaded worker): two simultaneously-missing granules must share
        exactly ONE inline write-back delegate. The unguarded lazy init
        constructed two; the loser's pending chunk maps were never drained
        by finish_granule, so its manifest went silently sparse."""
        import threading

        self._stub_read_module(monkeypatch, with_seam=True)

        constructed = []
        barrier = threading.Barrier(2)

        class _SlowInlineIndex:
            """Stands in for zagg.index.inline.InlineIndex. The barrier
            parks the first constructor, so a second thread that also
            reaches the ``_inline is None`` branch (the unguarded race)
            provably lands inside ``__init__`` too and both constructions
            are recorded. Under the lock exactly one thread enters; its
            barrier times out, breaks, and construction proceeds alone."""

            def __init__(self, write_back=False, store=None):
                try:
                    barrier.wait(timeout=1.0)
                except threading.BrokenBarrierError:
                    pass
                constructed.append(self)
                self.read_calls = []
                self.finished = []

            def read_group(self, h5obj, group, ds, shard_key, grid, arrow=False):
                self.read_calls.append(str(h5obj.resource))
                return "built-sentinel"

            def finish_granule(self, h5obj, granule_url):
                self.finished.append(granule_url)

        inline_mod = types.ModuleType("zagg.index.inline")
        inline_mod.InlineIndex = _SlowInlineIndex
        monkeypatch.setitem(sys.modules, "zagg.index.inline", inline_mod)

        gids = ["ATL03_GRANULE_A_007_01", "ATL03_GRANULE_B_007_01"]
        urls = [f"s3://bucket/{g}.h5" for g in gids]
        h5objs = [types.SimpleNamespace(resource=u) for u in urls]
        backend = sidecar_cls(store=str(tmp_path), on_miss="build")
        for g in gids:
            backend._cache[g] = None  # the store covers neither granule

        ds = {"read_plan": {"spatial_index": "segments"}}
        results, errors = [None, None], []

        def read(i):
            try:
                results[i] = backend.read_group(h5objs[i], "gt1l", ds, 1, grid=None)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=read, args=(i,)) for i in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert results == ["built-sentinel", "built-sentinel"]
        # Exactly one delegate, and both threads read through that same one.
        assert len(constructed) == 1
        delegate = constructed[0]
        assert backend._inline is delegate
        assert sorted(delegate.read_calls) == sorted(urls)
        # finish_granule forwards to the shared delegate for both granules.
        for h5, url in zip(h5objs, urls):
            backend.finish_granule(h5, url)
        assert sorted(delegate.finished) == sorted(urls)

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

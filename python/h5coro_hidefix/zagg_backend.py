"""The ``sidecar`` zagg index backend (zagg issue #160, phase 3).

Entry-point target for zagg's virtual-index registry::

    [project.entry-points."zagg.index_backends"]
    sidecar = "h5coro_hidefix.zagg_backend:SidecarIndex"

Implements the :class:`zagg.index.VirtualIndex` protocol as merged in
englacial/zagg PR #163 — this implementation is written against protocol
head ``87b941ed29618ba9b1dfee1ec2668392cd1f9ac3`` (merged to zagg main in
``dca9a91``); the pinned surface is ``read_group(h5obj, group, data_source,
shard_key, grid, arrow=False, granule_url=None)``, ``finish_granule(h5obj,
granule_url)``, the ``config_keys``/``required_config_keys`` validation
hooks, and the ``inline`` backend's write-back manifest schema
(``MANIFEST_DTYPES``) consumed here via :mod:`h5coro_hidefix.manifest`.

How a group read works
----------------------
Selection is the shared planned route (coarse geolocation read +
``plan_read`` — identical to ``hierarchical``/``inline``); only base-rate
*addressing* changes, through ``_planned_read_group``'s ``read_fn`` seam:

1. the granule's manifest ``<store>/<granule_id>.parquet`` is fetched once
   per granule and reconstructed into a decode-capable
   :class:`h5coro_hidefix.Index` via ``Index.from_chunks`` (no granule
   metadata I/O, no B-tree walk);
2. each planned slice asks the index for its chunk ranges (``read_plan``),
   fetches exactly those byte ranges through the worker's own credentialed
   h5coro driver (``h5obj.ioRequest`` — no second credential path), and
   decodes them with ``read_from_buffers`` (byte-identical to h5py/h5coro
   semantics, and immune to h5coro's chunk-aligned-start B-tree off-by-one
   since h5coro's B-tree is never consulted).

Granule identity: the worker only passes ``granule_url`` on the a-priori
arm (which is mutually exclusive with this backend), so the manifest key is
derived from ``h5obj.resource`` — the rewritten URL the worker opened the
granule with; its basename stem equals the granule id under every driver
rewrite, matching ``inline``'s write-back key convention.

``on_miss`` (this backend's policy for a granule the store does not cover,
or a dataset absent from its manifest): ``fallback`` (default) delegates
that group read to the hierarchical path; ``error`` raises (the worker
counts it as a read error); ``build`` delegates to an internal
``inline``-with-write-back backend, so served granules also populate the
store (zagg issue #160's deployment progression).

Concurrency contract (zagg PR #183): granule reads may interleave across
worker threads — the worker keeps K granules in flight on a bounded pool,
calling ``read_group`` from pool threads. Per-granule state must be keyed
by granule id (full resource URL), and ``finish_granule(h5obj,
granule_url)`` is called exactly once per granule, from the thread that
read it. Accordingly, ``_cache`` is granule-id-keyed (single-key get/pop
per granule, safe under the GIL) and the ``on_miss: build`` delegate is
constructed under a lock. Version note: ``on_miss: build`` with concurrent
granules requires zagg >= the release carrying zagg PR #183 (its
``_pending`` re-key to the full resource URL); older zagg's path-keyed
``_pending`` would interleave chunk maps across granules in the shared
inline delegate.

Imports: this module imports zagg (it subclasses the protocol) and lazily
imports pandas for parquet decoding — both are guaranteed in any
environment that discovers the entry point (zagg's own). The wheel's
install requirements stay numpy-only; ``import h5coro_hidefix`` never pulls
this module.
"""

from __future__ import annotations

import logging
import threading
from pathlib import PurePosixPath
from urllib.parse import urlsplit

from inspect import signature as _sig

from zagg.index import VirtualIndex

from h5coro_hidefix import Index
from h5coro_hidefix.manifest import datasets_from_manifest

logger = logging.getLogger(__name__)

_ON_MISS = ("fallback", "error", "build")

# Process-global in-flight range-GET budget. Each granule's read_fn pools its
# chunk fetches (_fetch_workers wide) and zagg multiplies that by its
# dataset-level read_workers pool and, under zagg PR #183, by K granules in
# flight -- K x read_workers x fetch-width GETs (~384 at K=6 with defaults)
# against the S3 driver's ~100-connection urllib3 pool, where the overflow
# queues until the 5s timeout fires and pool exhaustion masquerades as
# read_errors. The shared semaphore makes the queueing happen HERE instead:
# orderly, no timeout artifacts, and budget idled by a draining granule flows
# to granules still reading (tail-granule skew). Default 64 ~= the S3
# driver's max_pool_connections=100 with headroom; a single number,
# deliberately not auto-tuned (zagg's fleet A/B at K in {1,2,4,6} produces
# the evidence for a better default if one exists). Override without a
# release via env var ZAGG_HIDEFIX_FETCH_BUDGET (positive integer; parsed
# defensively -- anything else falls back to the default).
_FETCH_BUDGET = 64


def _fetch_budget() -> int:
    import os

    raw = os.environ.get("ZAGG_HIDEFIX_FETCH_BUDGET")
    if raw is not None:
        try:
            v = int(raw)
        except ValueError:
            v = 0
        if v >= 1:
            return v
        logger.warning(
            f"ignoring ZAGG_HIDEFIX_FETCH_BUDGET={raw!r} (need a positive "
            f"integer); using default {_FETCH_BUDGET}"
        )
    return _FETCH_BUDGET


_FETCH_SEMAPHORE = threading.BoundedSemaphore(_fetch_budget())


class _ManifestMiss(Exception):
    """Store has no manifest for this granule, or no rows for a dataset."""


def _granule_id(resource) -> str:
    """Granule id from the h5coro resource path (URL basename minus extension)."""
    return PurePosixPath(urlsplit(str(resource)).path).stem


def _fetch_workers(data_source) -> int:
    """Chunk-fetch pool width: ``data_source.read_workers`` (zagg issue #170).

    One knob governs both fan-out levels -- zagg pools across datasets with
    the same key, this backend pools the per-chunk fetches inside each read.
    zagg >= 0.15 validates the value at submission, and
    ``validate_index_config`` re-checks it at backend resolution; the guard
    here is the last line of defense (note it raises inside a group read,
    where zagg's worker counts it as a read error rather than aborting the
    shard). Default 8; ``1`` is serial (the pre-pool behavior).
    """
    w = (data_source or {}).get("read_workers", 8)
    if isinstance(w, bool) or not isinstance(w, int) or w < 1:
        raise ValueError(f"data_source.read_workers must be an integer >= 1 (got {w!r})")
    return w


def _fetch_chunks(h5obj, addrs, sizes, workers: int):
    """Fetch covering-chunk byte ranges, pooled and order-preserving.

    Each entry is one blocking ranged read through the worker's h5coro
    driver (``ioRequest(caching=False)`` -- no second credential path).
    ``ThreadPoolExecutor.map`` preserves input order and re-raises the first
    failure. A driver that swallows an error and returns ``None`` surfaces
    as ``OSError`` here -- transient I/O, never misdiagnosed as a decode
    failure inside ``read_from_buffers`` (zagg PR #173 review lesson).

    Sizing note: nominal fan-out is zagg's dataset-level pool times this
    width, and under zagg PR #183 also times the K granules in flight -- so
    every GET additionally acquires the process-global ``_FETCH_SEMAPHORE``
    (``_FETCH_BUDGET``, env-overridable via ZAGG_HIDEFIX_FETCH_BUDGET),
    which caps actual in-flight GETs regardless of how the three pool
    widths multiply. h5coro's S3Driver provisions its boto3 client with
    max_pool_connections=100, so budgeted concurrency is realized inside
    the driver, not queued in urllib3; its adaptive retries + 5s timeout
    degrade a failed range to None, which surfaces as OSError here. The
    semaphore is held only around the ioRequest call itself -- never while
    holding another lock -- so contention cannot deadlock.
    Caveat: h5coro's HTTPDriver shares one requests.Session across threads
    (officially unsupported sharing; risk concentrates in EDL redirect
    cookie-jar merges) -- the S3 driver is the deployed path; https callers
    wanting strict isolation can set read_workers: 1.
    """
    def fetch(i: int):
        with _FETCH_SEMAPHORE:
            buf = h5obj.ioRequest(int(addrs[i]), int(sizes[i]), caching=False)
        if buf is None:
            raise OSError(
                f"ranged read failed at {int(addrs[i])}+{int(sizes[i])} "
                "(driver returned None)"
            )
        return buf

    n = len(addrs)
    if workers <= 1 or n <= 1:
        return [fetch(i) for i in range(n)]
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=min(workers, n)) as pool:
        return list(pool.map(fetch, range(n)))


class SidecarIndex(VirtualIndex):
    """Selection via the planned route; addressing via the sidecar store."""

    name = "sidecar"
    config_keys = frozenset({"store", "on_miss"})
    required_config_keys = frozenset({"store"})

    def __init__(self, store: str, on_miss: str = "fallback"):
        self.store = store
        self.on_miss = on_miss
        # Per-granule state, keyed by granule id (see the concurrency
        # contract in the module docstring — granule reads may interleave
        # across worker threads): granule id -> reconstructed Index, or None
        # when the store had no manifest. Single-key get/pop per granule, so
        # safe under the GIL as-is.
        self._cache: dict[str, Index | None] = {}
        self._inline = None  # lazily-built inline+write_back delegate (on_miss: build)
        self._inline_lock = threading.Lock()  # guards _inline's one-time construction

    # -- config hooks (validate_index_config has already enforced key sets) --

    @classmethod
    def validate_index_config(cls, index_cfg: dict, data_source: dict | None = None) -> None:
        store = index_cfg.get("store")
        if not (isinstance(store, str) and store):
            raise ValueError(
                "index backend 'sidecar' requires 'store' "
                "(a local directory or s3://bucket/prefix)"
            )
        on_miss = index_cfg.get("on_miss", "fallback")
        if on_miss not in _ON_MISS:
            raise ValueError(f"index.on_miss must be one of {list(_ON_MISS)} (got {on_miss!r})")
        if data_source is not None:
            # Both read routes are served (mirrors zagg 0.15's inline): sources
            # with read_plan.spatial_index take the planned route, read-plan-less
            # (flat) sources the compiled full-read route -- no spatial_index
            # requirement anymore. The a-priori arm stays mutually exclusive.
            rp = data_source.get("read_plan")
            if isinstance(rp, dict) and "chunk_boundaries" in rp:
                raise ValueError(
                    "index backend 'sidecar' and read_plan.chunk_boundaries (the "
                    "a-priori arm) are mutually exclusive; drop one of them"
                )
            # Same gate zagg >=0.15 applies at submission -- validated here too
            # so hand-rolled worker payloads and older-zagg callers are rejected
            # at backend resolution rather than inside a per-group read (where
            # zagg's worker would swallow it into read_errors).
            w = data_source.get("read_workers")
            if w is not None and (isinstance(w, bool) or not isinstance(w, int) or w < 1):
                raise ValueError(
                    f"data_source.read_workers must be an integer >= 1 (got {w!r})"
                )

    @classmethod
    def from_index_config(cls, index_cfg: dict) -> "SidecarIndex":
        return cls(store=index_cfg["store"], on_miss=index_cfg.get("on_miss", "fallback"))

    # -- store access ---------------------------------------------------------

    def _fetch_manifest(self, granule_id: str):
        """Fetch + parse ``<store>/<granule_id>.parquet``; None when absent."""
        import io

        import obstore
        from zagg.store import open_object_store

        try:
            resp = obstore.get(open_object_store(self.store), f"{granule_id}.parquet")
            buf = bytes(resp.bytes())
        except FileNotFoundError:
            # obstore's NotFoundError subclasses FileNotFoundError (local + s3).
            return None
        import pandas as pd  # zagg-env guaranteed; engine: pyarrow or fastparquet

        return pd.read_parquet(io.BytesIO(buf))

    def _index_for(self, h5obj) -> Index | None:
        gid = _granule_id(h5obj.resource)
        if gid not in self._cache:
            df = self._fetch_manifest(gid)
            if df is None:
                self._cache[gid] = None
                logger.info(f"  sidecar: no manifest for {gid} (on_miss={self.on_miss})")
            else:
                specs = datasets_from_manifest(df)
                self._cache[gid] = Index.from_chunks(str(h5obj.resource), specs)
        return self._cache[gid]

    # -- addressing seam ------------------------------------------------------

    def _read_fn_for(self, vidx: Index, h5obj, workers: int = 8):
        """Buffer-fed reader for ``execute_read_plan``: (path, hyperslice) ->
        array, fetching chunk ranges through the worker's h5coro driver.

        Chunk fetches are pooled ``workers`` wide (zagg issue #170): the
        covering chunks of one planned read are independent ranged GETs, and
        their round trips -- not decode -- dominate the read wall on dense
        shards, which the dataset-level pool upstream in zagg cannot reach
        (it only overlaps *different* datasets' reads). Order is preserved,
        so ``read_from_buffers`` sees exactly the ``read_plan`` order.
        """
        import numpy as np

        def _read(path, start, end):
            addrs, sizes, _ = vidx.read_plan(path, start, end)
            buffers = _fetch_chunks(h5obj, addrs, sizes, workers)
            return vidx.read_from_buffers(path, buffers, start, end)

        def read_fn(path, hyperslice=None):
            try:
                if hyperslice is None:
                    return _read(path, None, None)
                parts = [_read(path, s, e) for s, e in hyperslice]
            except KeyError:
                # Dataset not covered by the manifest (lazy store coverage).
                raise _ManifestMiss(path) from None
            return parts[0] if len(parts) == 1 else np.concatenate(parts)

        return read_fn

    # -- protocol -------------------------------------------------------------

    def _miss(self, h5obj, group, data_source, shard_key, grid, arrow, granule_url, why):
        if self.on_miss == "error":
            raise FileNotFoundError(
                f"sidecar store {self.store!r} does not cover {why} (on_miss: error)"
            )
        if self.on_miss == "build":
            if self._inline is None:
                # Double-checked: read_group may run concurrently across
                # granule threads (see the module docstring's concurrency
                # contract); construct the shared delegate exactly once.
                # The import stays lazy so module load never pulls
                # zagg.index.inline.
                with self._inline_lock:
                    if self._inline is None:
                        from zagg.index.inline import InlineIndex

                        self._inline = InlineIndex(write_back=True, store=self.store)
            return self._inline.read_group(
                h5obj, group, data_source, shard_key, grid, arrow=arrow
            )
        # fallback: today's hierarchical path for this group read. Resolved
        # through the package namespace at call time (same monkeypatch
        # contract as HierarchicalIndex).
        import zagg.processing as _processing

        kwargs = {"arrow": arrow}
        if granule_url is not None:
            kwargs["granule_url"] = granule_url
        return _processing._read_group(h5obj, group, data_source, shard_key, grid, **kwargs)

    def read_group(self, h5obj, group, data_source, shard_key, grid, arrow=False, granule_url=None):
        from zagg.processing.read import _planned_read_group, _validate_planned_config

        # Two routes, one addressing seam (mirrors zagg 0.15's inline): planned
        # (chunk-aligned hyperslices) with a spatial index, compiled full-read
        # without one. The full-read seam (read_fn on _read_group_full) exists
        # from zagg 0.15.0; older zagg keeps the planned-only contract.
        rp = data_source.get("read_plan")
        planned = isinstance(rp, dict) and bool(rp.get("spatial_index"))
        if planned:
            _validate_planned_config(data_source)
            route = _planned_read_group
        else:
            try:
                from zagg.processing.read import _read_group_full as route
            except ImportError:
                route = None
            if route is None or "read_fn" not in str(_sig(route)):
                raise ValueError(
                    "index backend 'sidecar' on a read-plan-less (flat) data source "
                    "requires zagg >= 0.15 (the compiled full-read seam); either "
                    "upgrade zagg or add data_source.read_plan.spatial_index"
                )

        vidx = self._index_for(h5obj)
        if vidx is None:
            return self._miss(
                h5obj, group, data_source, shard_key, grid, arrow, granule_url,
                f"granule {_granule_id(h5obj.resource)!r}",
            )
        try:
            return route(
                h5obj, group, data_source, shard_key, grid,
                arrow=arrow,
                read_fn=self._read_fn_for(vidx, h5obj, workers=_fetch_workers(data_source)),
            )
        except _ManifestMiss as e:
            return self._miss(
                h5obj, group, data_source, shard_key, grid, arrow, granule_url,
                f"dataset {e.args[0]!r} of granule {_granule_id(h5obj.resource)!r}",
            )

    def finish_granule(self, h5obj, granule_url: str) -> None:
        """Drop the granule's reconstructed index; forward the write-back seam.

        Called exactly once per granule, from the thread that read it (the
        module docstring's concurrency contract) — granule reads may
        interleave, so the forwarded write-back seam relies on the shared
        inline delegate keying its pending state by granule id.
        """
        self._cache.pop(_granule_id(h5obj.resource), None)
        self._cache.pop(_granule_id(granule_url), None)
        if self._inline is not None:
            self._inline.finish_granule(h5obj, granule_url)

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

Imports: this module imports zagg (it subclasses the protocol) and lazily
imports pandas for parquet decoding — both are guaranteed in any
environment that discovers the entry point (zagg's own). The wheel's
install requirements stay numpy-only; ``import h5coro_hidefix`` never pulls
this module.
"""

from __future__ import annotations

import logging
from pathlib import PurePosixPath
from urllib.parse import urlsplit

from zagg.index import VirtualIndex

from h5coro_hidefix import Index
from h5coro_hidefix.manifest import datasets_from_manifest

logger = logging.getLogger(__name__)

_ON_MISS = ("fallback", "error", "build")


class _ManifestMiss(Exception):
    """Store has no manifest for this granule, or no rows for a dataset."""


def _granule_id(resource) -> str:
    """Granule id from the h5coro resource path (URL basename minus extension)."""
    return PurePosixPath(urlsplit(str(resource)).path).stem


class SidecarIndex(VirtualIndex):
    """Selection via the planned route; addressing via the sidecar store."""

    name = "sidecar"
    config_keys = frozenset({"store", "on_miss"})
    required_config_keys = frozenset({"store"})

    def __init__(self, store: str, on_miss: str = "fallback"):
        self.store = store
        self.on_miss = on_miss
        # Per-granule state (the worker reads granules serially and calls
        # ``finish_granule`` after each): granule id -> reconstructed Index,
        # or None when the store had no manifest.
        self._cache: dict[str, Index | None] = {}
        self._inline = None  # lazily-built inline+write_back delegate (on_miss: build)

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
            rp = data_source.get("read_plan")
            if not (isinstance(rp, dict) and rp.get("spatial_index")):
                raise ValueError(
                    "index backend 'sidecar' requires data_source.read_plan.spatial_index "
                    "(chunk addressing plugs into the planned read path)"
                )
            if "chunk_boundaries" in rp:
                raise ValueError(
                    "index backend 'sidecar' and read_plan.chunk_boundaries (the "
                    "a-priori arm) are mutually exclusive; drop one of them"
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

    def _read_fn_for(self, vidx: Index, h5obj):
        """Buffer-fed reader for ``execute_read_plan``: (path, hyperslice) ->
        array, fetching chunk ranges through the worker's h5coro driver."""
        import numpy as np

        def _read(path, start, end):
            addrs, sizes, _ = vidx.read_plan(path, start, end)
            buffers = [
                h5obj.ioRequest(int(a), int(s), caching=False)
                for a, s in zip(addrs, sizes)
            ]
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

        rp = data_source.get("read_plan")
        if not (isinstance(rp, dict) and rp.get("spatial_index")):
            raise ValueError("index backend 'sidecar' requires data_source.read_plan.spatial_index")
        _validate_planned_config(data_source)

        vidx = self._index_for(h5obj)
        if vidx is None:
            return self._miss(
                h5obj, group, data_source, shard_key, grid, arrow, granule_url,
                f"granule {_granule_id(h5obj.resource)!r}",
            )
        try:
            return _planned_read_group(
                h5obj, group, data_source, shard_key, grid,
                arrow=arrow, read_fn=self._read_fn_for(vidx, h5obj),
            )
        except _ManifestMiss as e:
            return self._miss(
                h5obj, group, data_source, shard_key, grid, arrow, granule_url,
                f"dataset {e.args[0]!r} of granule {_granule_id(h5obj.resource)!r}",
            )

    def finish_granule(self, h5obj, granule_url: str) -> None:
        """Drop the granule's reconstructed index; forward the write-back seam."""
        self._cache.pop(_granule_id(h5obj.resource), None)
        self._cache.pop(_granule_id(granule_url), None)
        if self._inline is not None:
            self._inline.finish_granule(h5obj, granule_url)

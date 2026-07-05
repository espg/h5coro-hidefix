# h5coro-hidefix

Compiled companion to [h5coro](https://github.com/SlideRuleEarth/h5coro): a
tiny [pyo3](https://pyo3.rs) binding over the
[hidefix](https://github.com/gauteh/hidefix) crate (consumed from crates.io —
no hidefix source lives in this tree). hidefix reads chunked HDF5 files
through a pre-built chunk index instead of the HDF5 library's B-tree walk,
decoding chunks in parallel (gzip + shuffle), which benchmarked ~6× faster
than pure-Python h5coro on ICESat-2 ATL03 workloads — and ~15–20× once the
index is built ahead of time and reloaded.

This package exists because the upstream `hidefix` Python binding cannot
save, load, or enumerate its index, and squeezes length-1 dimensions on read.
This binding fixes all three while keeping the surface minimal:

- `Index(path)` — build an index from a local HDF5 file (~0.2 s / ~0.6 MB for
  a ~2 GB ATL03 granule).
- `Index.save(path)` / `Index.load(path, source=None)` — persist / restore
  the index (bincode via hidefix's public serde impls) so reads need zero
  HDF5 metadata I/O. `source` points reads at the data file when it lives
  somewhere else than at index-build time.
- `Index.datasets()` — full dataset paths, recursing groups.
- `Index.chunks(dataset)` — the chunk table as numpy arrays
  `(addrs, sizes, offsets)`: file byte offset, stored (compressed) byte
  count, and dataspace coordinates per chunk.
- `Index.read(dataset, start=None, end=None)` — rows `[start, end)` along
  dim 0 (remaining dims in full) as a numpy array with native dtype and the
  exact request shape. **Never squeezes**: a `(1, 5)` request returns
  `(1, 5)`, byte-identical with `h5py_dataset[start:end]`. The GIL is
  released for the duration of the read.
- `Index.read_plan(dataset, start=None, end=None)` /
  `Index.read_from_buffers(dataset, buffers, start=None, end=None)` — the
  object-store read path: the plan lists the chunks a read needs (the same
  `(addrs, sizes, offsets)` triple as `chunks()`, restricted to the row
  range, ascending dataspace offset); the caller fetches those byte ranges
  itself and hands the raw stored bytes back for decode + assembly,
  byte-identical to `read()`. No file or network access happens inside the
  binding.
- `Index.from_chunks(source, datasets)` — construct a decode-capable index
  from an **external chunk manifest** (e.g. rows from a parquet sidecar
  store): no granule file access, no stored bincode. `Index.filters(dataset)`
  (`{"gzip": int|None, "shuffle": bool, "byte_order": ...}`) completes the
  extraction side, so `Index(path)` → getters → `from_chunks` round-trips
  byte-identically.
- **zagg `sidecar` backend** (`h5coro_hidefix.zagg_backend.SidecarIndex`) —
  registered under the `zagg.index_backends` entry-point group, so a zagg
  environment with this wheel installed discovers it automatically:

  ```yaml
  data_source:
    index:
      backend: sidecar
      store: s3://bucket/zagg-index/ATL03/007/
      on_miss: fallback      # fallback | error | build
  ```

  Selection uses zagg's shared planned route; addressing reconstructs the
  granule's index from `<store>/<granule_id>.parquet` (zagg's inline
  write-back manifest schema) via `from_chunks`, fetches exactly the needed
  chunk ranges through the worker's own credentialed h5coro driver, and
  decodes with `read_from_buffers`. Written against the zagg virtual-index
  protocol at englacial/zagg PR #163 head `87b941e` (see the module
  docstring). Importing `h5coro_hidefix` alone never pulls zagg/pandas —
  the backend module is only imported by zagg's entry-point discovery.

Design discussion: [englacial/zagg#155](https://github.com/englacial/zagg/issues/155),
[englacial/zagg#160](https://github.com/englacial/zagg/issues/160)
(parquet-primary sidecar store).

## Install

```sh
pip install h5coro-hidefix
```

Or from source (needs a Rust toolchain and `cmake` — the bundled static
libhdf5 is built at compile time; no system HDF5 required at build *or* run
time):

```sh
pip install .
```

## Quickstart

```python
import h5coro_hidefix as hx

idx = hx.Index("ATL03_20190105163308_01260202_007_01.h5")   # ~0.2 s

idx.datasets()                        # ['/gt1l/heights/h_ph', ...]
idx.shape("/gt1l/heights/signal_conf_ph")   # (n_photons, 5)
idx.dtype("/gt1l/heights/h_ph")             # 'float64'

# persist the index; later reads skip all HDF5 metadata I/O
idx.save("granule.idx")
idx = hx.Index.load("granule.idx", source="ATL03_...h5")

arr = idx.read("/gt1l/heights/h_ph", 1_000_000, 2_000_000)  # float64 (1000000,)
one = idx.read("/gt1l/heights/signal_conf_ph", 7, 8)        # int8 (1, 5) -- not (5,)

addrs, sizes, offsets = idx.chunks("/gt1l/heights/h_ph")    # uint64 arrays
```

## Obstore-fed reads (the Lambda worker flow)

Workers never open the HDF5 file: indices are built once at catalog time,
and at read time the worker loads the index, asks which byte ranges a
row-slice needs, fetches those ranges itself (obstore/boto3 ranged GETs),
and hands the raw bytes back for decode:

```python
import obstore
import h5coro_hidefix as hx

idx = hx.Index.load("granule.idx")          # ~1 ms; zero HDF5 metadata I/O
name = "/gt1l/heights/h_ph"

addrs, sizes, _ = idx.read_plan(name, row0, row1)
buffers = obstore.get_ranges(               # caller owns the fetch policy:
    store, "ATL03_...h5",                   # coalescing, concurrency, retries
    starts=addrs.tolist(),
    ends=(addrs + sizes).tolist(),
)
arr = idx.read_from_buffers(name, buffers, row0, row1)
# arr is byte-identical to idx.read(name, row0, row1) on a local copy
```

`buffers` must line up 1:1, in order, with `read_plan`'s chunks; any
bytes-like objects work (`bytes` is zero-copy, others are copied once).
`ValueError` is raised on a wrong buffer count or a buffer whose length
differs from the chunk's stored size; the GIL is released during decode.
This path removes any dependency on hidefix-side S3 support.

## Manifest-built indices: `from_chunks` (parquet-primary store)

When the chunk manifest lives in a durable store (parquet sidecars — see
englacial/zagg#160), the index is reconstructed from it on demand; the
granule file is never opened and no bincode is ever stored:

```python
# caller side: read the manifest (pyarrow / obstore) -- this package stays
# numpy-only at runtime, so parquet decoding happens outside it
idx = hx.Index.from_chunks("s3-key-or-local-path.h5", {
    "/gt1l/heights/h_ph": dict(
        dtype="<f8",              # numpy dtype str; byte order honored
        shape=(23_692_855,),
        chunk_shape=(100_000,),
        gzip=6,                   # deflate level, bool, or None (see below)
        shuffle=True,
        addrs=addrs,              # u64[k]   chunk byte offsets in the file
        sizes=sizes,              # u64[k]   stored (compressed) byte counts
        offsets=offsets,          # u64[k, ndim] dataspace chunk coordinates
        filter_mask=masks,        # optional; must be all zero (see below)
    ),
    ...
})
arr = idx.read_from_buffers(name, buffers, row0, row1)  # as above
```

Chunk rows need not be pre-sorted. The result is a real index — `read`,
`read_plan`, `read_from_buffers`, `save`, `chunks` all behave exactly as if
it had been built from the file.

**Metadata contract for extractors** (what a manifest must carry, per
dataset): `dtype`, `shape`, `chunk_shape`, `gzip` (**level int | bool |
null** — manifests that cannot see the deflate level, e.g. h5coro's
metadata parse, may emit a boolean: decode only checks presence, so `True`
records a placeholder level), `shuffle`, plus the chunk table (`addrs`,
`sizes`, and per-chunk dataspace `offsets` — for 1-D data the offset equals
the element start index; for N-D it must be stored explicitly). All of it is available from a real index via `datasets()`,
`shape()`, `chunk_shape()`, `dtype()`, `filters()` and `chunks()`.

**Per-chunk filter masks are unrepresentable**: hidefix models filters at
dataset level (`shuffle`, `gzip`), and its chunk records carry only
`(addr, size, offset)`. A nonzero HDF5 `filter_mask` (bit i = filter i
skipped for that chunk) therefore raises `ValueError` in `from_chunks`.
ATL03 masks are all zero (verified in englacial/zagg PR #159).

## Scope and limitations

- The binding itself performs only local-file I/O (`read()`); object-store
  reads are the caller's job via `read_plan`/`read_from_buffers` above.
- Filters: gzip (deflate) + shuffle + byte-order — full coverage for ATL03
  and most NASA Earthdata HDF5. Other filters (szip, lzf, scaleoffset) fail
  at index time.
- Hyperslabs are row ranges on dimension 0; remaining dimensions are read in
  full.
- The serialized index format is hidefix's bincode encoding of its Rust
  types — treat it as an opaque cache keyed by (file, hidefix version), not
  as a stable interchange format.

## Licensing

The binding code in this repository is MIT. Published wheels statically
embed third-party components; their license texts ship inside the wheel
(`*.dist-info/licenses/`):

- **hidefix** — MIT, Copyright 2023 Gaute Hope (`LICENSE-hidefix`). The
  earlier upstream metadata ambiguity is resolved: MIT was confirmed in
  [gauteh/hidefix#48](https://github.com/gauteh/hidefix/issues/48) and the
  stale `Cargo.toml` field fixed in
  [gauteh/hidefix#49](https://github.com/gauteh/hidefix/pull/49).
- **HDF5**, built and bundled by `hdf5-metno-src` — The HDF Group's
  BSD-style license (`LICENSE-hdf5`).

Remaining statically-linked Rust dependencies are MIT/Apache-2.0
dual-licensed or similarly permissive.

## Development

```sh
python -m venv .venv && . .venv/bin/activate
pip install maturin pytest h5py numpy
maturin develop --release
pytest -v          # hermetic tests generate their own HDF5 files
```

A second, local-only test tier runs automatically when real ATL03 granules
are present (default `~/ignore/zagg_neon_atl03_test_shard/granules`, override
with `H5CORO_HIDEFIX_GRANULE_DIR`).

CI builds wheels for manylinux x86_64 + aarch64 (static libhdf5, `cmake`
installed in the manylinux image) and macOS arm64, abi3 (`>=3.9`), one wheel
per platform. The publish job triggers on version tags and publishes to PyPI
via Trusted Publishing.

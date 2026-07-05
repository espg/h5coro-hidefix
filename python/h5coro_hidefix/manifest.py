"""Sidecar manifest -> ``Index.from_chunks()`` input (zagg-free).

The manifest is zagg's granule-keyed write-back parquet (one row per HDF5
chunk; schema = ``zagg.index.inline.MANIFEST_DTYPES`` at englacial/zagg
``87b941ed29618ba9b1dfee1ec2668392cd1f9ac3``, PR #163): PR #159's offsets
columns (``dataset``, ``chunk_idx``, ``elem_start``, ``elem_end``,
``byte_offset``, ``nbytes``, ``filter_mask``) plus the per-dataset decode
metadata ``from_chunks`` needs — ``chunk_offset`` / ``shape`` /
``chunk_shape`` as JSON-encoded lists, ``dtype`` as the byte-order-explicit
``np.dtype(...).str`` form (``<f8``, ``|i1``), and ``gzip`` as a *boolean*
(filter presence; the deflate level is invisible to h5coro's metadata parse
and irrelevant for decode — ``from_chunks`` maps the bool).

This module is deliberately free of zagg/pandas/pyarrow imports: it consumes
any column mapping (a dict of sequences, a ``pandas.DataFrame``, ...) so the
parquet decoding stays caller-side and the hermetic tests need no zagg.
"""

import json

__all__ = ["MANIFEST_COLUMNS", "datasets_from_manifest"]

#: Columns a manifest must carry (zagg's MANIFEST_DTYPES keys, pinned above).
MANIFEST_COLUMNS = (
    "dataset",
    "chunk_idx",
    "elem_start",
    "elem_end",
    "byte_offset",
    "nbytes",
    "filter_mask",
    "chunk_offset",
    "dtype",
    "shape",
    "chunk_shape",
    "gzip",
    "shuffle",
)


def datasets_from_manifest(columns):
    """Convert manifest columns into the ``Index.from_chunks`` datasets dict.

    ``columns`` maps column name -> sequence (all the same length); the
    required names are :data:`MANIFEST_COLUMNS` minus ``elem_start`` /
    ``elem_end`` (derivable, unused here). Rows are grouped by ``dataset``
    and ordered by ``chunk_idx``. Datasets whose ``dtype`` is empty (types
    h5coro cannot map — strings, compounds) are skipped: they cannot be
    decoded, so reads of them surface as not-covered.

    Raises ``KeyError`` on a missing column and ``ValueError`` on malformed
    JSON cells; everything else (chunk-count/offset/filter-mask validation)
    is ``Index.from_chunks``'s job.
    """
    for col in ("dataset", "chunk_idx", "byte_offset", "nbytes", "filter_mask"):
        if col not in columns:
            raise KeyError(f"manifest is missing required column '{col}'")

    rows_by_ds = {}
    for i, name in enumerate(columns["dataset"]):
        rows_by_ds.setdefault(str(name), []).append(i)

    out = {}
    for name, rows in rows_by_ds.items():
        rows.sort(key=lambda i: int(columns["chunk_idx"][i]))
        first = rows[0]
        dtype = str(columns["dtype"][first])
        if not dtype:
            continue
        try:
            shape = json.loads(columns["shape"][first])
            chunk_shape = json.loads(columns["chunk_shape"][first])
            offsets = [json.loads(columns["chunk_offset"][i]) for i in rows]
        except (TypeError, json.JSONDecodeError) as e:
            raise ValueError(f"manifest dataset {name}: malformed JSON cell: {e}") from e
        out[name] = dict(
            dtype=dtype,
            shape=shape,
            chunk_shape=chunk_shape,
            gzip=bool(columns["gzip"][first]),
            shuffle=bool(columns["shuffle"][first]),
            addrs=[int(columns["byte_offset"][i]) for i in rows],
            sizes=[int(columns["nbytes"][i]) for i in rows],
            offsets=offsets,
            filter_mask=[int(columns["filter_mask"][i]) for i in rows],
        )
    return out

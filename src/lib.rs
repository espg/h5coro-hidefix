//! h5coro-hidefix: compiled companion to h5coro.
//!
//! A thin pyo3 binding over the [hidefix](https://crates.io/crates/hidefix)
//! crate (consumed from crates.io -- no source vendored here). It exposes
//! what the upstream `hidefix` Python binding lacks:
//!
//! 1. index save/load (bincode, via hidefix's public serde impls),
//! 2. chunk enumeration (`(addr, size, offset...)` per chunk),
//! 3. reads with h5py-compatible semantics: a row-range hyperslab on dim 0
//!    never squeezes -- a `(1, 5)` request returns shape `(1, 5)`,
//! 4. construction from an external chunk manifest (`Index.from_chunks`),
//!    so a parquet-primary sidecar store needs no granule file access and
//!    no durable bincode.

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::anyhow;
use hidefix::filters::byteorder::{Order, ToNative};
use hidefix::idx::{self, DatasetD, DatasetExt, Datatype};
use hidefix::prelude::{ParReaderExt, ReaderExt};
use hidefix::reader::cache::CacheReader;
use numpy::{PyArray1, PyArrayDyn, PyArrayMethods};
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyTuple};

/// Owns a deserialized index together with the buffer it may borrow from.
///
/// hidefix's index deserialization is zero-copy: the per-dataset chunk tables
/// (`Cow<[Chunk<D>]>`) borrow directly from the serialized bytes.
struct Holder {
    /// May borrow from `_buf`. Declared first so it is dropped before `_buf`
    /// (struct fields drop in declaration order).
    idx: idx::Index<'static>,
    /// Backing bytes for a loaded index; `None` for a freshly built index.
    /// Never mutated, and the heap allocation address is stable across moves
    /// of the box, so borrows into it stay valid for the holder's lifetime.
    _buf: Option<Box<[u8]>>,
}

/// Deserialize an index out of an owned buffer, keeping the buffer alive
/// alongside it (zero-copy chunk tables). Shared by `load()` (bytes from
/// disk) and `from_chunks()` (bytes from the in-memory shim).
fn holder_from_bytes(buf: Box<[u8]>, what: &str) -> anyhow::Result<Holder> {
    // SAFETY: `idx` borrows from the heap allocation behind `buf`. That
    // allocation's address is stable across moves of the box, the buffer is
    // never mutated, and `Holder`'s field order guarantees `idx` is dropped
    // before `buf`. The 'static lifetime never escapes `Holder`.
    let slice: &'static [u8] = unsafe { std::slice::from_raw_parts(buf.as_ptr(), buf.len()) };
    let idx: idx::Index<'static> =
        bincode::deserialize(slice).map_err(|e| anyhow!("cannot deserialize index {what}: {e}"))?;
    Ok(Holder {
        idx,
        _buf: Some(buf),
    })
}

/// Serialize-side mirrors of hidefix's private `Index`/`GroupIndex` shells.
///
/// `Index`/`GroupIndex` cannot be constructed from parts (private fields; the
/// only constructors walk an open HDF5 file), but their bincode layout is
/// just field order and types. These shims match that layout exactly --
/// (path, root) and (path, datasets, groups) -- while the `DatasetD` payload
/// is built with hidefix's *public* API (`Dataset::new`, `Chunk::new`) and
/// serialized by hidefix's own serde impls. Serializing a shim and
/// deserializing through the `load()` path yields a real `Index` without any
/// file access; the bincode bytes never touch disk.
#[derive(serde::Serialize)]
struct GroupShim {
    path: Option<PathBuf>,
    datasets: HashMap<String, DatasetD<'static>>,
    groups: HashMap<String, GroupShim>,
}

impl GroupShim {
    fn new(path: Option<PathBuf>) -> Self {
        Self {
            path,
            datasets: HashMap::new(),
            groups: HashMap::new(),
        }
    }
}

#[derive(serde::Serialize)]
struct IndexShim {
    path: Option<PathBuf>,
    root: GroupShim,
}

/// Parse a numpy dtype string (`<f8`, `|i1`, `>u4`, `f8`, or names like
/// `float64`) into hidefix's datatype + byte order.
fn parse_dtype(s: &str) -> Option<(Datatype, Order)> {
    let (order, rest) = match s.as_bytes().first()? {
        b'<' => (Order::LE, &s[1..]),
        b'>' => (Order::BE, &s[1..]),
        b'=' | b'|' => (Order::native(), &s[1..]),
        _ => (Order::native(), s),
    };
    let dt = match rest {
        "f4" | "float32" => Datatype::Float(4),
        "f8" | "float64" => Datatype::Float(8),
        "i1" | "int8" => Datatype::Int(1),
        "i2" | "int16" => Datatype::Int(2),
        "i4" | "int32" => Datatype::Int(4),
        "i8" | "int64" => Datatype::Int(8),
        "u1" | "uint8" => Datatype::UInt(1),
        "u2" | "uint16" => Datatype::UInt(2),
        "u4" | "uint32" => Datatype::UInt(4),
        "u8" | "uint64" => Datatype::UInt(8),
        _ => return None,
    };
    Some((dt, order))
}

/// Everything `from_chunks` needs for one dataset, extracted from its dict.
struct DsSpec {
    dtype: Datatype,
    order: Order,
    shape: Vec<u64>,
    chunk_shape: Vec<u64>,
    shuffle: bool,
    gzip: Option<u8>,
    addrs: Vec<u64>,
    sizes: Vec<u64>,
    offsets: Vec<Vec<u64>>,
}

fn spec_err(name: &str, msg: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(format!("dataset {name}: {msg}"))
}

fn req<'py, T: FromPyObject<'py>>(d: &Bound<'py, PyDict>, name: &str, key: &str) -> PyResult<T> {
    d.get_item(key)?
        .filter(|v| !v.is_none())
        .ok_or_else(|| spec_err(name, format!("missing required key '{key}'")))?
        .extract::<T>()
        .map_err(|e| spec_err(name, format!("invalid '{key}': {e}")))
}

fn parse_spec(name: &str, d: &Bound<'_, PyDict>) -> PyResult<DsSpec> {
    let dtype_s: String = req(d, name, "dtype")?;
    let (dtype, order) = parse_dtype(&dtype_s)
        .ok_or_else(|| spec_err(name, format!("unsupported dtype '{dtype_s}'")))?;
    let shape: Vec<u64> = req(d, name, "shape")?;
    let chunk_shape: Vec<u64> = req(d, name, "chunk_shape")?;
    let shuffle: bool = req(d, name, "shuffle")?;
    let gzip_item = d
        .get_item("gzip")?
        .ok_or_else(|| spec_err(name, "missing required key 'gzip'"))?;
    let gzip: Option<u8> = if gzip_item.is_none() {
        None
    } else if gzip_item.is_instance_of::<PyBool>() {
        // Manifests that cannot see the deflate level (e.g. zagg write-back
        // via h5coro's metadata parse) emit a boolean; decode only checks
        // presence, so `True` stores a placeholder level (6) that is not
        // meaningful. Must be tested before the int branch: bool is a
        // Python int subclass.
        if gzip_item.extract::<bool>()? {
            Some(6)
        } else {
            None
        }
    } else {
        let level: u8 = gzip_item.extract().map_err(|_| {
            spec_err(
                name,
                "'gzip' must be a deflate level (0-9), a bool, or None",
            )
        })?;
        if level > 9 {
            return Err(spec_err(name, format!("invalid deflate level {level}")));
        }
        Some(level)
    };
    let addrs: Vec<u64> = req(d, name, "addrs")?;
    let sizes: Vec<u64> = req(d, name, "sizes")?;
    let offsets: Vec<Vec<u64>> = req(d, name, "offsets")?;

    let ndim = shape.len();
    if ndim == 0 || ndim > 9 {
        return Err(spec_err(name, format!("rank {ndim} unsupported (1-9)")));
    }
    if chunk_shape.len() != ndim {
        return Err(spec_err(
            name,
            format!(
                "chunk_shape rank {} != shape rank {ndim}",
                chunk_shape.len()
            ),
        ));
    }
    if chunk_shape.contains(&0) {
        return Err(spec_err(name, "chunk_shape entries must be nonzero"));
    }
    let k = addrs.len();
    if sizes.len() != k || offsets.len() != k {
        return Err(spec_err(
            name,
            format!(
                "chunk table length mismatch: addrs {k}, sizes {}, offsets {}",
                sizes.len(),
                offsets.len()
            ),
        ));
    }
    if let Some(fm_item) = d.get_item("filter_mask")? {
        if !fm_item.is_none() {
            let masks: Vec<u64> = fm_item
                .extract()
                .map_err(|e| spec_err(name, format!("invalid 'filter_mask': {e}")))?;
            if masks.len() != k {
                return Err(spec_err(
                    name,
                    format!("filter_mask length {} != chunk count {k}", masks.len()),
                ));
            }
            if let Some(i) = masks.iter().position(|&m| m != 0) {
                return Err(spec_err(
                    name,
                    format!(
                        "chunk {i} has nonzero filter_mask {}; hidefix's chunk model \
                         cannot represent per-chunk filter masks",
                        masks[i]
                    ),
                ));
            }
        }
    }
    Ok(DsSpec {
        dtype,
        order,
        shape,
        chunk_shape,
        shuffle,
        gzip,
        addrs,
        sizes,
        offsets,
    })
}

/// Build the concrete `Dataset<'static, D>` from a validated spec.
fn build_dataset_d<const D: usize>(
    name: &str,
    spec: &DsSpec,
) -> PyResult<idx::Dataset<'static, D>> {
    let shape: [u64; D] = spec.shape.as_slice().try_into().expect("rank checked");
    let chunk_shape: [u64; D] = spec
        .chunk_shape
        .as_slice()
        .try_into()
        .expect("rank checked");
    let mut chunks: Vec<idx::Chunk<D>> = Vec::with_capacity(spec.addrs.len());
    for (i, ((&addr, &size), off)) in spec
        .addrs
        .iter()
        .zip(&spec.sizes)
        .zip(&spec.offsets)
        .enumerate()
    {
        let off: [u64; D] = off.as_slice().try_into().map_err(|_| {
            spec_err(
                name,
                format!("offsets row {i} has {} entries, expected {D}", off.len()),
            )
        })?;
        for d in 0..D {
            if !off[d].is_multiple_of(chunk_shape[d]) || (shape[d] > 0 && off[d] >= shape[d]) {
                return Err(spec_err(
                    name,
                    format!(
                        "chunk {i} offset {off:?} is not a chunk-aligned position \
                         inside shape {shape:?} (chunk_shape {chunk_shape:?})"
                    ),
                ));
            }
        }
        chunks.push(idx::Chunk::new(addr, size, off));
    }
    chunks.sort();
    if let Some(w) = chunks.windows(2).find(|w| w[0].cmp(&w[1]).is_eq()) {
        return Err(spec_err(
            name,
            format!("duplicate chunk offset {:?}", w[0].offset_u64()),
        ));
    }
    let expected: u64 = shape
        .iter()
        .zip(&chunk_shape)
        .map(|(s, c)| s.div_ceil(*c))
        .product();
    if chunks.len() as u64 != expected {
        return Err(spec_err(
            name,
            format!(
                "chunk table has {} rows but shape {shape:?} / chunk_shape \
                 {chunk_shape:?} requires {expected}",
                chunks.len()
            ),
        ));
    }
    idx::Dataset::new(
        spec.dtype,
        spec.order,
        shape,
        chunks,
        chunk_shape,
        spec.shuffle,
        spec.gzip,
    )
    .map_err(|e| spec_err(name, e))
}

fn build_dataset(name: &str, spec: &DsSpec) -> PyResult<DatasetD<'static>> {
    Ok(match spec.shape.len() {
        1 => DatasetD::D1(build_dataset_d::<1>(name, spec)?),
        2 => DatasetD::D2(build_dataset_d::<2>(name, spec)?),
        3 => DatasetD::D3(build_dataset_d::<3>(name, spec)?),
        4 => DatasetD::D4(build_dataset_d::<4>(name, spec)?),
        5 => DatasetD::D5(build_dataset_d::<5>(name, spec)?),
        6 => DatasetD::D6(build_dataset_d::<6>(name, spec)?),
        7 => DatasetD::D7(build_dataset_d::<7>(name, spec)?),
        8 => DatasetD::D8(build_dataset_d::<8>(name, spec)?),
        9 => DatasetD::D9(build_dataset_d::<9>(name, spec)?),
        n => return Err(spec_err(name, format!("rank {n} unsupported (1-9)"))),
    })
}

/// Insert a dataset at its full path, creating intermediate group shims.
fn insert_dataset(
    root: &mut GroupShim,
    source: &Path,
    path: &str,
    dsd: DatasetD<'static>,
) -> PyResult<()> {
    let mut parts: Vec<&str> = path.trim_start_matches('/').split('/').collect();
    let ds_name = match parts.pop() {
        Some(n) if !n.is_empty() => n,
        _ => {
            return Err(PyValueError::new_err(format!(
                "invalid dataset path: '{path}'"
            )))
        }
    };
    let mut group = root;
    for part in parts {
        if part.is_empty() {
            return Err(PyValueError::new_err(format!(
                "invalid dataset path: '{path}'"
            )));
        }
        group = group
            .groups
            .entry(part.to_string())
            .or_insert_with(|| GroupShim::new(Some(source.to_path_buf())));
    }
    if group.datasets.insert(ds_name.to_string(), dsd).is_some() {
        return Err(PyValueError::new_err(format!(
            "duplicate dataset path: '{path}'"
        )));
    }
    Ok(())
}

/// Run `$body` with `$ds` bound to the concrete `Dataset<'_, D>` inside a
/// `DatasetD`, for every dimensionality variant.
macro_rules! with_dataset {
    ($dsd:expr, $ds:ident => $body:expr) => {
        match $dsd {
            DatasetD::D0($ds) => $body,
            DatasetD::D1($ds) => $body,
            DatasetD::D2($ds) => $body,
            DatasetD::D3($ds) => $body,
            DatasetD::D4($ds) => $body,
            DatasetD::D5($ds) => $body,
            DatasetD::D6($ds) => $body,
            DatasetD::D7($ds) => $body,
            DatasetD::D8($ds) => $body,
            DatasetD::D9($ds) => $body,
        }
    };
}

fn dtype_str(dtype: Datatype) -> PyResult<&'static str> {
    Ok(match dtype {
        Datatype::UInt(1) => "uint8",
        Datatype::UInt(2) => "uint16",
        Datatype::UInt(4) => "uint32",
        Datatype::UInt(8) => "uint64",
        Datatype::Int(1) => "int8",
        Datatype::Int(2) => "int16",
        Datatype::Int(4) => "int32",
        Datatype::Int(8) => "int64",
        Datatype::Float(4) => "float32",
        Datatype::Float(8) => "float64",
        dt => {
            return Err(PyValueError::new_err(format!(
                "unsupported datatype: {dt:?}"
            )))
        }
    })
}

fn walk_datasets(group: &idx::GroupIndex, prefix: &str, out: &mut Vec<String>) {
    for name in group.datasets().keys() {
        out.push(format!("{prefix}/{name}"));
    }
    for (name, sub) in group.groups() {
        walk_datasets(sub, &format!("{prefix}/{name}"), out);
    }
}

fn to_py_tuple(py: Python<'_>, vals: &[u64]) -> PyResult<Py<PyTuple>> {
    Ok(PyTuple::new(py, vals.iter().copied())?.unbind())
}

/// An in-memory, sparse "file" serving caller-provided chunk buffers at their
/// original in-file addresses. Handing this to hidefix's own `CacheReader`
/// makes buffer-fed decode take the exact decode + placement path `read()`
/// takes against a local file.
struct SegmentReader<'a> {
    /// (addr, bytes), sorted by addr.
    segments: Vec<(u64, &'a [u8])>,
    pos: u64,
}

impl<'a> SegmentReader<'a> {
    fn new(mut segments: Vec<(u64, &'a [u8])>) -> Self {
        segments.sort_by_key(|(addr, _)| *addr);
        Self { segments, pos: 0 }
    }
}

impl Read for SegmentReader<'_> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let outside = || {
            std::io::Error::other(format!(
                "read at file offset {} is outside the provided chunk buffers",
                self.pos
            ))
        };
        let i = self
            .segments
            .partition_point(|(addr, bytes)| addr + bytes.len() as u64 <= self.pos);
        let (addr, bytes) = self.segments.get(i).copied().ok_or_else(outside)?;
        if self.pos < addr {
            return Err(outside());
        }
        let off = (self.pos - addr) as usize;
        let n = std::cmp::min(buf.len(), bytes.len() - off);
        buf[..n].copy_from_slice(&bytes[off..off + n]);
        self.pos += n as u64;
        Ok(n)
    }
}

impl Seek for SegmentReader<'_> {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        self.pos = match pos {
            SeekFrom::Start(p) => p,
            SeekFrom::Current(d) => self
                .pos
                .checked_add_signed(d)
                .ok_or_else(|| std::io::Error::other("seek out of range"))?,
            SeekFrom::End(_) => {
                return Err(std::io::Error::other("SeekFrom::End is unsupported"));
            }
        };
        Ok(self.pos)
    }
}

/// Normalize a sequence of bytes-like objects to `bytes`: zero-copy for
/// `bytes` items, otherwise a copy via `bytes(memoryview(item))` (bytearray,
/// memoryview, obstore Bytes, ...). `memoryview` rejects non-buffer objects.
fn extract_buffers<'py>(
    py: Python<'py>,
    buffers: &Bound<'py, PyAny>,
) -> PyResult<Vec<Bound<'py, PyBytes>>> {
    let memoryview = py.import("builtins")?.getattr("memoryview")?;
    let mut out = Vec::new();
    for item in buffers.try_iter()? {
        let item = item?;
        out.push(match item.downcast_into::<PyBytes>() {
            Ok(b) => b,
            Err(e) => {
                let mv = memoryview.call1((e.into_inner(),)).map_err(|err| {
                    PyValueError::new_err(format!("buffer {} is not bytes-like: {err}", out.len()))
                })?;
                mv.call_method0("tobytes")?.downcast_into::<PyBytes>()?
            }
        });
    }
    Ok(out)
}

/// A serializable HDF5 chunk index with h5py-compatible hyperslab reads.
///
/// Build with ``Index(path)`` from a local HDF5 file, persist with
/// ``Index.save(path)``, and restore with ``Index.load(path, source=...)``
/// without touching the HDF5 file again.
#[pyclass(frozen, module = "h5coro_hidefix")]
struct Index {
    holder: Arc<Holder>,
    source: Option<PathBuf>,
}

impl Index {
    fn dataset<'a>(&'a self, name: &str) -> PyResult<&'a DatasetD<'a>> {
        self.holder
            .idx
            .dataset(name)
            .ok_or_else(|| PyKeyError::new_err(format!("no such dataset: {name}")))
    }

    /// Validate and normalize a dim-0 row range, shared by read/read_plan/
    /// read_from_buffers. Returns (start, end, full shape).
    fn row_range(
        ds: &DatasetD<'_>,
        start: Option<u64>,
        end: Option<u64>,
    ) -> PyResult<(u64, u64, Vec<u64>)> {
        let shape = DatasetExt::shape(ds).to_vec();
        if shape.is_empty() {
            return Err(PyValueError::new_err(
                "0-dimensional datasets are not supported",
            ));
        }
        let start = start.unwrap_or(0);
        let end = end.unwrap_or(shape[0]);
        if start > end || end > shape[0] {
            return Err(PyValueError::new_err(format!(
                "invalid row range [{start}, {end}) for dim-0 size {}",
                shape[0]
            )));
        }
        Ok((start, end, shape))
    }

    /// Indices into the dataset's chunk table (ascending dataspace offset,
    /// same order as chunks()) of the chunks intersecting rows [start, end).
    fn plan_indices(ds: &DatasetD<'_>, start: u64, end: u64) -> Vec<usize> {
        if start == end {
            return Vec::new();
        }
        with_dataset!(ds, d => {
            // .first() rather than [0] keeps the (unreachable) D0 macro arm
            // compiling; callers guard 0-d out via row_range().
            let c0 = d.chunk_shape.first().copied().unwrap_or(1);
            d.chunks
                .iter()
                .enumerate()
                .filter(|(_, c)| {
                    let o0 = c.offset.first().map(|o| o.get()).unwrap_or(0);
                    o0 < end && start < o0 + c0
                })
                .map(|(i, _)| i)
                .collect()
        })
    }

    /// (addrs, sizes, offsets) numpy triple for the whole chunk table, or the
    /// `filter`ed subset (in `filter` order).
    fn chunk_arrays(
        py: Python<'_>,
        dsd: &DatasetD<'_>,
        filter: Option<&[usize]>,
    ) -> PyResult<Py<PyAny>> {
        let (addrs, sizes, offsets, ndim) = with_dataset!(dsd, ds => {
            let ndim = ds.shape.len();
            let pick: Vec<usize> = match filter {
                Some(f) => f.to_vec(),
                None => (0..ds.chunks.len()).collect(),
            };
            let mut addrs = Vec::with_capacity(pick.len());
            let mut sizes = Vec::with_capacity(pick.len());
            let mut offsets = Vec::with_capacity(pick.len() * ndim);
            for &i in &pick {
                let c = &ds.chunks[i];
                addrs.push(c.addr.get());
                sizes.push(c.size.get());
                offsets.extend(c.offset.iter().map(|o| o.get()));
            }
            (addrs, sizes, offsets, ndim)
        });
        let n = addrs.len();
        let addrs = PyArray1::from_vec(py, addrs);
        let sizes = PyArray1::from_vec(py, sizes);
        let offsets = PyArray1::from_vec(py, offsets).reshape([n, ndim])?;
        Ok((addrs, sizes, offsets)
            .into_pyobject(py)?
            .into_any()
            .unbind())
    }

    /// Decode caller-provided chunk buffers and assemble the request, using
    /// hidefix's own `CacheReader` over a `SegmentReader` so the result is
    /// byte-identical to `read()`.
    fn assemble_typed<T>(
        py: Python<'_>,
        dsd: &DatasetD<'_>,
        segments: Vec<(u64, &[u8])>,
        indices: &[u64],
        counts: &[u64],
    ) -> PyResult<Py<PyAny>>
    where
        T: numpy::Element + byte_slice_cast::ToMutByteSlice + Send,
        [T]: ToNative,
    {
        let dims: Vec<usize> = counts.iter().map(|&c| c as usize).collect();
        let arr = unsafe { PyArrayDyn::<T>::new(py, dims, false) };
        let n_expected: u64 = counts.iter().product();
        if n_expected == 0 {
            return Ok(arr.into_any().unbind());
        }
        let dst = unsafe { arr.as_slice_mut() }
            .map_err(|e| PyValueError::new_err(format!("array not contiguous: {e}")))?;
        py.allow_threads(|| -> anyhow::Result<()> {
            let fd = SegmentReader::new(segments);
            let n = with_dataset!(dsd, ds => {
                let mut r = CacheReader::with_dataset(ds, fd)?;
                r.values_to((indices, counts), dst)? as u64
            });
            let expected_bytes = n_expected * std::mem::size_of::<T>() as u64;
            if n != expected_bytes {
                return Err(anyhow!(
                    "short read: got {n} bytes, expected {expected_bytes}"
                ));
            }
            Ok(())
        })?;
        Ok(arr.into_any().unbind())
    }

    fn read_typed<T>(
        &self,
        py: Python<'_>,
        ds: &DatasetD<'_>,
        indices: &[u64],
        counts: &[u64],
    ) -> PyResult<Py<PyAny>>
    where
        T: numpy::Element + byte_slice_cast::ToMutByteSlice + Send,
        [T]: ToNative,
    {
        let dims: Vec<usize> = counts.iter().map(|&c| c as usize).collect();
        // Allocate the full, never-squeezed request shape up front.
        let arr = unsafe { PyArrayDyn::<T>::new(py, dims, false) };
        let n_expected: u64 = counts.iter().product();
        if n_expected == 0 {
            return Ok(arr.into_any().unbind());
        }
        let dst = unsafe { arr.as_slice_mut() }
            .map_err(|e| PyValueError::new_err(format!("array not contiguous: {e}")))?;
        let source = self.source.as_ref().ok_or_else(|| {
            PyValueError::new_err(
                "no source HDF5 file associated with this index; \
                 pass `source=` to Index.load()",
            )
        })?;
        py.allow_threads(|| -> anyhow::Result<()> {
            let r = ds.as_par_reader(&source.as_path())?;
            // values_to_par returns the number of *bytes* written into dst.
            let n = r.values_to_par((indices, counts), dst)? as u64;
            let expected_bytes = n_expected * std::mem::size_of::<T>() as u64;
            if n != expected_bytes {
                return Err(anyhow!(
                    "short read: got {n} bytes, expected {expected_bytes}"
                ));
            }
            Ok(())
        })?;
        Ok(arr.into_any().unbind())
    }
}

#[pymethods]
impl Index {
    /// Index(path)
    ///
    /// Build a chunk index by walking a local HDF5 file's metadata.
    #[new]
    fn new(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        let idx = py.allow_threads(|| idx::Index::index(&path))?;
        Ok(Self {
            holder: Arc::new(Holder { idx, _buf: None }),
            source: Some(path),
        })
    }

    /// load(path, source=None)
    ///
    /// Load a previously saved index (bincode). ``source`` overrides the HDF5
    /// file path used for reads; it defaults to the path recorded when the
    /// index was built, which is only valid on the machine that built it.
    #[staticmethod]
    #[pyo3(signature = (path, source=None))]
    fn load(py: Python<'_>, path: PathBuf, source: Option<PathBuf>) -> PyResult<Self> {
        let (holder, embedded) = py.allow_threads(|| -> anyhow::Result<_> {
            let buf = std::fs::read(&path)?.into_boxed_slice();
            let holder = holder_from_bytes(buf, &path.display().to_string())?;
            let embedded = holder.idx.path().map(|p| p.to_path_buf());
            Ok((holder, embedded))
        })?;
        Ok(Self {
            holder: Arc::new(holder),
            source: source.or(embedded),
        })
    }

    /// from_chunks(source, datasets)
    ///
    /// Construct a decode-capable index from an externally-supplied chunk
    /// manifest -- no granule file access, no stored bincode. ``source`` is
    /// the path reads resolve against (may not exist locally when only
    /// ``read_plan``/``read_from_buffers`` are used). ``datasets`` maps full
    /// dataset paths to dicts with keys: ``dtype`` (numpy str), ``shape``,
    /// ``chunk_shape``, ``gzip`` (deflate level, bool, or None -- a bare
    /// ``True`` records presence with a placeholder level, since decode
    /// only checks presence), ``shuffle``
    /// (bool), ``addrs`` (u64[k]), ``sizes`` (u64[k]), ``offsets``
    /// (u64[k, ndim]), and optionally ``filter_mask`` (must be all zero).
    /// Chunk rows need not be pre-sorted. The result behaves exactly like an
    /// index built from the file: ``read``/``read_plan``/
    /// ``read_from_buffers``/``save``/``chunks`` all work unchanged.
    #[staticmethod]
    fn from_chunks(source: PathBuf, datasets: &Bound<'_, PyDict>) -> PyResult<Self> {
        let mut root = GroupShim::new(Some(source.clone()));
        for (key, value) in datasets.iter() {
            let name: String = key
                .extract()
                .map_err(|_| PyValueError::new_err("dataset keys must be str paths"))?;
            let spec_dict = value.downcast::<PyDict>().map_err(|_| {
                PyValueError::new_err(format!("dataset {name}: value must be a dict"))
            })?;
            let spec = parse_spec(&name, spec_dict)?;
            let dsd = build_dataset(&name, &spec)?;
            insert_dataset(&mut root, &source, &name, dsd)?;
        }
        let shim = IndexShim {
            path: Some(source.clone()),
            root,
        };
        let bytes = bincode::serialize(&shim)
            .map_err(|e| PyValueError::new_err(format!("cannot encode index: {e}")))?;
        let holder = holder_from_bytes(bytes.into_boxed_slice(), "built by from_chunks")?;
        Ok(Self {
            holder: Arc::new(holder),
            source: Some(source),
        })
    }

    /// save(path)
    ///
    /// Serialize the index to ``path`` (bincode, hidefix's public serde
    /// impls; the same format the ``hfxidx`` CLI writes).
    fn save(&self, py: Python<'_>, path: PathBuf) -> PyResult<()> {
        let holder = self.holder.clone();
        py.allow_threads(move || -> anyhow::Result<()> {
            let bytes = bincode::serialize(&holder.idx)?;
            std::fs::write(&path, bytes)?;
            Ok(())
        })?;
        Ok(())
    }

    /// The HDF5 file reads resolve against (None if unknown; see load()).
    #[getter]
    fn source(&self) -> Option<PathBuf> {
        self.source.clone()
    }

    /// datasets()
    ///
    /// Sorted full paths (e.g. ``/gt1l/heights/h_ph``) of every indexed
    /// dataset, recursing through groups.
    fn datasets(&self) -> Vec<String> {
        let mut out = Vec::new();
        walk_datasets(&self.holder.idx, "", &mut out);
        out.sort();
        out
    }

    /// shape(dataset) -> tuple[int, ...]
    fn shape(&self, py: Python<'_>, dataset: &str) -> PyResult<Py<PyTuple>> {
        to_py_tuple(py, DatasetExt::shape(self.dataset(dataset)?))
    }

    /// chunk_shape(dataset) -> tuple[int, ...]
    fn chunk_shape(&self, py: Python<'_>, dataset: &str) -> PyResult<Py<PyTuple>> {
        to_py_tuple(py, DatasetExt::chunk_shape(self.dataset(dataset)?))
    }

    /// dtype(dataset) -> str
    ///
    /// Numpy dtype name (native byte order after read), e.g. ``float64``.
    fn dtype(&self, dataset: &str) -> PyResult<&'static str> {
        dtype_str(self.dataset(dataset)?.dtype())
    }

    /// filters(dataset) -> dict
    ///
    /// The dataset's filter chain and storage byte order:
    /// ``{"gzip": int | None, "shuffle": bool, "byte_order": "little" |
    /// "big" | "unknown"}``. Together with ``datasets()``, ``shape()``,
    /// ``chunk_shape()``, ``dtype()`` and ``chunks()`` this is everything an
    /// extractor needs to produce ``from_chunks()`` input from a real index.
    fn filters(&self, py: Python<'_>, dataset: &str) -> PyResult<Py<PyAny>> {
        let dsd = self.dataset(dataset)?;
        let (gzip, shuffle, order) = with_dataset!(dsd, ds => (ds.gzip, ds.shuffle, ds.order));
        let d = PyDict::new(py);
        d.set_item("gzip", gzip)?;
        d.set_item("shuffle", shuffle)?;
        d.set_item(
            "byte_order",
            match order {
                Order::BE => "big",
                Order::LE => "little",
                Order::Unknown => "unknown",
            },
        )?;
        Ok(d.into_any().unbind())
    }

    /// chunks(dataset) -> (addrs, sizes, offsets)
    ///
    /// The dataset's chunk table, sorted by dataspace offset:
    /// ``addrs`` uint64 (n,) byte offsets in the HDF5 file, ``sizes`` uint64
    /// (n,) stored (compressed) byte counts, ``offsets`` uint64 (n, ndim)
    /// dataspace coordinates where each chunk begins.
    fn chunks(&self, py: Python<'_>, dataset: &str) -> PyResult<Py<PyAny>> {
        Self::chunk_arrays(py, self.dataset(dataset)?, None)
    }

    /// read_plan(dataset, start=None, end=None) -> (addrs, sizes, offsets)
    ///
    /// The chunks that ``read(dataset, start, end)`` needs, as the same
    /// numpy triple ``chunks()`` returns, restricted to the chunks whose
    /// dim-0 extent intersects rows ``[start, end)`` and ordered by
    /// ascending dataspace offset. Fetch bytes ``[addr, addr + size)`` per
    /// chunk (e.g. S3 ranged GETs) and pass them, in this exact order, to
    /// ``read_from_buffers()``.
    #[pyo3(signature = (dataset, start=None, end=None))]
    fn read_plan(
        &self,
        py: Python<'_>,
        dataset: &str,
        start: Option<u64>,
        end: Option<u64>,
    ) -> PyResult<Py<PyAny>> {
        let ds = self.dataset(dataset)?;
        let (start, end, _) = Self::row_range(ds, start, end)?;
        let plan = Self::plan_indices(ds, start, end);
        Self::chunk_arrays(py, ds, Some(&plan))
    }

    /// read_from_buffers(dataset, buffers, start=None, end=None) -> numpy.ndarray
    ///
    /// Decode caller-provided stored chunk bytes and assemble exactly what
    /// ``read(dataset, start, end)`` returns: same dtype, same shape, no
    /// squeeze, byte-identical. ``buffers`` is a sequence of bytes-like
    /// objects corresponding 1:1, in order, to the chunks returned by
    /// ``read_plan(dataset, start, end)``. No file access is performed --
    /// this is the object-store read path where the caller fetches the byte
    /// ranges itself. Raises ``ValueError`` on a wrong buffer count or a
    /// buffer whose length differs from the chunk's stored size. Releases
    /// the GIL during decode.
    #[pyo3(signature = (dataset, buffers, start=None, end=None))]
    fn read_from_buffers(
        &self,
        py: Python<'_>,
        dataset: &str,
        buffers: &Bound<'_, PyAny>,
        start: Option<u64>,
        end: Option<u64>,
    ) -> PyResult<Py<PyAny>> {
        let ds = self.dataset(dataset)?;
        let (start, end, shape) = Self::row_range(ds, start, end)?;
        let plan = Self::plan_indices(ds, start, end);
        let bufs = extract_buffers(py, buffers)?;
        if bufs.len() != plan.len() {
            return Err(PyValueError::new_err(format!(
                "expected {} buffers for rows [{start}, {end}) (read_plan order), got {}",
                plan.len(),
                bufs.len()
            )));
        }
        let segments = with_dataset!(ds, d => {
            let mut segments = Vec::with_capacity(plan.len());
            for (b, &i) in bufs.iter().zip(&plan) {
                let c = &d.chunks[i];
                let (got, want) = (b.as_bytes().len() as u64, c.size.get());
                if got != want {
                    return Err(PyValueError::new_err(format!(
                        "buffer {} has {got} bytes but the chunk at dataspace offset {:?} \
                         stores {want} bytes",
                        segments.len(),
                        c.offset_u64(),
                    )));
                }
                segments.push((c.addr.get(), b.as_bytes()));
            }
            segments
        });
        let mut indices = vec![0u64; shape.len()];
        indices[0] = start;
        let mut counts = shape;
        counts[0] = end - start;
        match ds.dtype() {
            Datatype::UInt(1) => Self::assemble_typed::<u8>(py, ds, segments, &indices, &counts),
            Datatype::UInt(2) => Self::assemble_typed::<u16>(py, ds, segments, &indices, &counts),
            Datatype::UInt(4) => Self::assemble_typed::<u32>(py, ds, segments, &indices, &counts),
            Datatype::UInt(8) => Self::assemble_typed::<u64>(py, ds, segments, &indices, &counts),
            Datatype::Int(1) => Self::assemble_typed::<i8>(py, ds, segments, &indices, &counts),
            Datatype::Int(2) => Self::assemble_typed::<i16>(py, ds, segments, &indices, &counts),
            Datatype::Int(4) => Self::assemble_typed::<i32>(py, ds, segments, &indices, &counts),
            Datatype::Int(8) => Self::assemble_typed::<i64>(py, ds, segments, &indices, &counts),
            Datatype::Float(4) => Self::assemble_typed::<f32>(py, ds, segments, &indices, &counts),
            Datatype::Float(8) => Self::assemble_typed::<f64>(py, ds, segments, &indices, &counts),
            dt => Err(PyValueError::new_err(format!(
                "unsupported datatype: {dt:?}"
            ))),
        }
    }

    /// read(dataset, start=None, end=None) -> numpy.ndarray
    ///
    /// Read rows ``[start, end)`` along dimension 0 (all remaining dimensions
    /// in full), defaulting to the whole dataset. The result has the exact
    /// request shape -- length-1 dimensions are never squeezed -- and native
    /// dtype, matching ``h5py_dataset[start:end]``. Releases the GIL for the
    /// duration of the read; chunks are decoded in parallel.
    #[pyo3(signature = (dataset, start=None, end=None))]
    fn read(
        &self,
        py: Python<'_>,
        dataset: &str,
        start: Option<u64>,
        end: Option<u64>,
    ) -> PyResult<Py<PyAny>> {
        let ds = self.dataset(dataset)?;
        let (start, end, shape) = Self::row_range(ds, start, end)?;
        let mut indices = vec![0u64; shape.len()];
        indices[0] = start;
        let mut counts = shape;
        counts[0] = end - start;
        match ds.dtype() {
            Datatype::UInt(1) => self.read_typed::<u8>(py, ds, &indices, &counts),
            Datatype::UInt(2) => self.read_typed::<u16>(py, ds, &indices, &counts),
            Datatype::UInt(4) => self.read_typed::<u32>(py, ds, &indices, &counts),
            Datatype::UInt(8) => self.read_typed::<u64>(py, ds, &indices, &counts),
            Datatype::Int(1) => self.read_typed::<i8>(py, ds, &indices, &counts),
            Datatype::Int(2) => self.read_typed::<i16>(py, ds, &indices, &counts),
            Datatype::Int(4) => self.read_typed::<i32>(py, ds, &indices, &counts),
            Datatype::Int(8) => self.read_typed::<i64>(py, ds, &indices, &counts),
            Datatype::Float(4) => self.read_typed::<f32>(py, ds, &indices, &counts),
            Datatype::Float(8) => self.read_typed::<f64>(py, ds, &indices, &counts),
            dt => Err(PyValueError::new_err(format!(
                "unsupported datatype: {dt:?}"
            ))),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Index(datasets: {}, source: {:?})",
            self.datasets().len(),
            self.source
        )
    }
}

#[pymodule]
#[pyo3(name = "_native")]
fn h5coro_hidefix(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Index>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

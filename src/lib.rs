//! h5coro-hidefix: compiled companion to h5coro.
//!
//! A thin pyo3 binding over the [hidefix](https://crates.io/crates/hidefix)
//! crate (consumed from crates.io -- no source vendored here). It exposes the
//! three things the upstream `hidefix` Python binding lacks:
//!
//! 1. index save/load (bincode, via hidefix's public serde impls),
//! 2. chunk enumeration (`(addr, size, offset...)` per chunk),
//! 3. reads with h5py-compatible semantics: a row-range hyperslab on dim 0
//!    never squeezes -- a `(1, 5)` request returns shape `(1, 5)`.

use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::anyhow;
use hidefix::filters::byteorder::ToNative;
use hidefix::idx::{self, DatasetD, DatasetExt, Datatype};
use hidefix::prelude::{ParReaderExt, ReaderExt};
use hidefix::reader::cache::CacheReader;
use numpy::{PyArray1, PyArrayDyn, PyArrayMethods};
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyTuple};

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
            // SAFETY: `idx` borrows from the heap allocation behind `buf`.
            // That allocation's address is stable across moves of the box, the
            // buffer is never mutated, and `Holder`'s field order guarantees
            // `idx` is dropped before `buf`. The 'static lifetime never
            // escapes `Holder`.
            let slice: &'static [u8] =
                unsafe { std::slice::from_raw_parts(buf.as_ptr(), buf.len()) };
            let idx: idx::Index<'static> = bincode::deserialize(slice)
                .map_err(|e| anyhow!("cannot deserialize index {}: {e}", path.display()))?;
            let embedded = idx.path().map(|p| p.to_path_buf());
            Ok((
                Holder {
                    idx,
                    _buf: Some(buf),
                },
                embedded,
            ))
        })?;
        Ok(Self {
            holder: Arc::new(holder),
            source: source.or(embedded),
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
fn h5coro_hidefix(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Index>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

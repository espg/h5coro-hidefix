import h5py
import numpy as np
import pytest

N = 10_000
CHUNK = 1_000


@pytest.fixture(scope="session")
def h5file(tmp_path_factory):
    """A small HDF5 file mimicking ATL03 shapes and filters.

    - 1-D float64, gzip+shuffle, chunked (like /gtXX/heights/h_ph)
    - 2-D (n, 5) int8, gzip+shuffle, chunked (like /gtXX/heights/signal_conf_ph)
    - a contiguous unfiltered dataset and a scalar, for coverage
    """
    path = tmp_path_factory.mktemp("data") / "atl03_like.h5"
    rng = np.random.default_rng(42)
    with h5py.File(path, "w") as f:
        heights = f.create_group("gt1l/heights")
        heights.create_dataset(
            "h_ph",
            data=rng.normal(scale=100.0, size=N),
            dtype="f8",
            chunks=(CHUNK,),
            compression="gzip",
            compression_opts=6,
            shuffle=True,
        )
        heights.create_dataset(
            "signal_conf_ph",
            data=rng.integers(-2, 5, size=(N, 5), dtype=np.int8),
            dtype="i1",
            chunks=(CHUNK, 5),
            compression="gzip",
            compression_opts=6,
            shuffle=True,
        )
        heights.create_dataset(
            "lat_ph",
            data=rng.uniform(-88, -60, size=N),
            dtype="f8",
            chunks=(CHUNK,),
            compression="gzip",
            compression_opts=6,
            shuffle=True,
        )
        f.create_dataset("meta/plain", data=np.arange(37, dtype="i4"))  # contiguous
        f.create_dataset("scalar", data=np.float64(3.5))  # 0-d
    return path


@pytest.fixture(scope="session")
def ref(h5file):
    """Reference arrays read with h5py."""
    out = {}
    with h5py.File(h5file, "r") as f:
        for name in (
            "/gt1l/heights/h_ph",
            "/gt1l/heights/signal_conf_ph",
            "/gt1l/heights/lat_ph",
            "/meta/plain",
        ):
            out[name] = f[name][...]
    return out

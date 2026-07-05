"""Integration tier: real zagg (>= PR #163 protocol) + this package installed.

Skipped unless zagg is importable. Mirrors englacial/zagg tests/test_index.py's
fixture conventions at protocol head 87b941ed29618ba9b1dfee1ec2668392cd1f9ac3:
the atl03_mini.h5 fixture (extracted from a zagg checkout when available),
the ATL03-shaped planned data_source, and the leaf-set grid stub. The money
test: an inline-write-back manifest served back by SidecarIndex is
row-identical to the hierarchical reference read.
"""

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

zagg_index = pytest.importorskip("zagg.index")

from zagg.index import available_index_backends, validate_index_config  # noqa: E402

ZAGG_REPO = Path(os.environ.get("ZAGG_REPO", Path.home() / "software" / "zagg"))
PROTOCOL_SHA = "87b941ed29618ba9b1dfee1ec2668392cd1f9ac3"


def test_entry_point_discovered():
    from h5coro_hidefix.zagg_backend import SidecarIndex

    backends = available_index_backends()
    assert backends.get("sidecar") is SidecarIndex


def test_config_validation_through_zagg():
    ds = {"read_plan": {"spatial_index": "segments"}}
    validate_index_config({"backend": "sidecar", "store": "/tmp/x"}, ds)
    with pytest.raises(ValueError, match="requires keys \\['store'\\]"):
        validate_index_config({"backend": "sidecar"}, ds)
    with pytest.raises(ValueError, match="not accepted"):
        validate_index_config({"backend": "sidecar", "store": "/x", "nope": 1}, ds)
    with pytest.raises(ValueError, match="on_miss"):
        validate_index_config({"backend": "sidecar", "store": "/x", "on_miss": "maybe"}, ds)


# ---------------------------------------------------------------------------
# End-to-end round trip on zagg's index fixture (needs the zagg repo checkout
# for tests/data/index/atl03_mini.h5 -- extracted via git show, tree untouched)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fixture_h5(tmp_path_factory):
    if not (ZAGG_REPO / ".git").exists():
        pytest.skip(f"no zagg checkout at {ZAGG_REPO} (set ZAGG_REPO)")
    out = tmp_path_factory.mktemp("zagg_fixture") / "atl03_mini.h5"
    ref = f"{PROTOCOL_SHA}:tests/data/index/atl03_mini.h5"
    proc = subprocess.run(
        ["git", "-C", str(ZAGG_REPO), "show", ref], capture_output=True
    )
    if proc.returncode != 0:
        pytest.skip(f"cannot extract {ref} from {ZAGG_REPO}")
    out.write_bytes(proc.stdout)
    return out


def _open_fixture(path):
    from h5coro import filedriver
    from h5coro import h5coro as h5c

    return h5c.H5Coro(str(path), filedriver.FileDriver, errorChecking=True, verbose=False)


def _fixture_data_source():
    """Copy of test_index.py's ATL03-shaped planned data_source (pinned SHA)."""
    return {
        "groups": ["gt1l", "gt2l"],
        "coordinates": {
            "latitude": "/{group}/heights/lat_ph",
            "longitude": "/{group}/heights/lon_ph",
        },
        "variables": {"h_ph": "/{group}/heights/h_ph"},
        "filters": [
            {"dataset": "/{group}/heights/signal_conf_ph", "column": 0, "op": "ne", "value": -2}
        ],
        "base_level": "photons",
        "levels": {
            "photons": {
                "path": "/{group}/heights",
                "coordinates": {"latitude": "lat_ph", "longitude": "lon_ph"},
                "link": None,
            },
            "segments": {
                "path": "/{group}/geolocation",
                "coordinates": {
                    "latitude": "reference_photon_lat",
                    "longitude": "reference_photon_lon",
                },
                "link": {
                    "to": "photons",
                    "index_beg": "/{group}/geolocation/ph_index_beg",
                    "count": "/{group}/geolocation/segment_ph_cnt",
                    "index_base": 1,
                },
            },
        },
        "read_plan": {"spatial_index": "segments", "pad": 1},
    }


class _LeafSetGrid:
    def __init__(self, leaves):
        self._leaves = np.asarray(sorted(leaves), dtype=np.int64)

    def assign(self, lats, lons):
        return np.round(np.asarray(lats)).astype(np.int64)

    def shards_of(self, leaf_ids):
        return np.isin(leaf_ids, self._leaves).astype(np.int64)


LEAVES = (4, 5, 13, 14, 104, 105)  # test_index.py's _UNALIGNED_LEAVES


def _read_all_groups(backend, h5obj, ds, grid):
    import pandas as pd

    frames = [backend.read_group(h5obj, g, ds, 1, grid) for g in ds["groups"]]
    frames = [f for f in frames if f is not None]
    return pd.concat(frames, ignore_index=True) if frames else None


@pytest.fixture(scope="module")
def store_with_manifest(fixture_h5, tmp_path_factory):
    """Populate a local sidecar store via zagg's own inline write-back."""
    from zagg.index.inline import InlineIndex

    store = tmp_path_factory.mktemp("sidecar_store")
    backend = InlineIndex(write_back=True, store=str(store))
    h5obj = _open_fixture(fixture_h5)
    ds = _fixture_data_source()
    grid = _LeafSetGrid(LEAVES)
    for g in ds["groups"]:
        backend.read_group(h5obj, g, ds, 1, grid)
    backend.finish_granule(h5obj, str(fixture_h5))
    assert (store / "atl03_mini.parquet").exists()
    return store


class TestSidecarRoundTrip:
    def test_row_identical_to_hierarchical(self, fixture_h5, store_with_manifest):
        import pandas as pd

        from h5coro_hidefix.zagg_backend import SidecarIndex
        from zagg.index.hierarchical import HierarchicalIndex

        ds = _fixture_data_source()
        grid = _LeafSetGrid(LEAVES)
        ref = _read_all_groups(HierarchicalIndex(), _open_fixture(fixture_h5), ds, grid)
        sidecar = SidecarIndex(store=str(store_with_manifest), on_miss="error")
        got = _read_all_groups(sidecar, _open_fixture(fixture_h5), ds, grid)
        assert ref is not None and got is not None
        pd.testing.assert_frame_equal(got, ref)

    def test_on_miss_error_raises(self, fixture_h5, tmp_path):
        from h5coro_hidefix.zagg_backend import SidecarIndex

        sidecar = SidecarIndex(store=str(tmp_path / "empty"), on_miss="error")
        with pytest.raises(FileNotFoundError, match="sidecar store"):
            sidecar.read_group(
                _open_fixture(fixture_h5), "gt1l", _fixture_data_source(), 1,
                _LeafSetGrid(LEAVES),
            )

    def test_on_miss_fallback_matches_hierarchical(self, fixture_h5, tmp_path):
        import pandas as pd

        from h5coro_hidefix.zagg_backend import SidecarIndex
        from zagg.index.hierarchical import HierarchicalIndex

        ds = _fixture_data_source()
        grid = _LeafSetGrid(LEAVES)
        ref = _read_all_groups(HierarchicalIndex(), _open_fixture(fixture_h5), ds, grid)
        sidecar = SidecarIndex(store=str(tmp_path / "empty"), on_miss="fallback")
        got = _read_all_groups(sidecar, _open_fixture(fixture_h5), ds, grid)
        pd.testing.assert_frame_equal(got, ref)

    def test_on_miss_build_populates_store_and_matches(self, fixture_h5, tmp_path):
        import pandas as pd

        from h5coro_hidefix.zagg_backend import SidecarIndex
        from zagg.index.hierarchical import HierarchicalIndex

        ds = _fixture_data_source()
        grid = _LeafSetGrid(LEAVES)
        store = tmp_path / "store"
        ref = _read_all_groups(HierarchicalIndex(), _open_fixture(fixture_h5), ds, grid)
        sidecar = SidecarIndex(store=str(store), on_miss="build")
        h5obj = _open_fixture(fixture_h5)
        got = _read_all_groups(sidecar, h5obj, ds, grid)
        pd.testing.assert_frame_equal(got, ref)
        sidecar.finish_granule(h5obj, str(fixture_h5))
        assert (store / "atl03_mini.parquet").exists()
        # second pass: now served from the store, still identical
        sidecar2 = SidecarIndex(store=str(store), on_miss="error")
        got2 = _read_all_groups(sidecar2, _open_fixture(fixture_h5), ds, grid)
        pd.testing.assert_frame_equal(got2, ref)

    def test_finish_granule_clears_cache(self, fixture_h5, store_with_manifest):
        from h5coro_hidefix.zagg_backend import SidecarIndex

        sidecar = SidecarIndex(store=str(store_with_manifest))
        h5obj = _open_fixture(fixture_h5)
        sidecar.read_group(h5obj, "gt1l", _fixture_data_source(), 1, _LeafSetGrid(LEAVES))
        assert sidecar._cache
        sidecar.finish_granule(h5obj, str(fixture_h5))
        assert not sidecar._cache

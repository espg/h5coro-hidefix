"""Minimal copy of zagg's VirtualIndex protocol for hermetic conformance tests.

PINNED to englacial/zagg PR #163 head 87b941ed29618ba9b1dfee1ec2668392cd1f9ac3
(``src/zagg/index/__init__.py``, merged to zagg main in dca9a91): class-level
``name`` / ``config_keys`` / ``required_config_keys``, the two classmethod
hooks, and the ``read_group`` / ``finish_granule`` signatures, verbatim minus
docstrings. If the protocol shifts upstream, refresh this stub AND
``h5coro_hidefix/zagg_backend.py`` together.

The hermetic tests install this under ``sys.modules['zagg.index']`` so the
entry-point target imports without a zagg environment.
"""

from typing import ClassVar


class VirtualIndex:
    name: ClassVar[str] = ""
    config_keys: ClassVar[frozenset] = frozenset()
    required_config_keys: ClassVar[frozenset] = frozenset()

    @classmethod
    def validate_index_config(cls, index_cfg: dict, data_source: dict | None = None) -> None:
        pass

    @classmethod
    def from_index_config(cls, index_cfg: dict) -> "VirtualIndex":
        return cls()

    def read_group(
        self,
        h5obj,
        group: str,
        data_source: dict,
        shard_key: int,
        grid,
        arrow: bool = False,
        granule_url: str | None = None,
    ):
        raise NotImplementedError

    def finish_granule(self, h5obj, granule_url: str) -> None:
        pass

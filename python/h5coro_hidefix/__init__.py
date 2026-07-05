"""h5coro-hidefix: compiled companion to h5coro.

The public surface is the compiled ``Index`` (see ``h5coro_hidefix._native``).
The zagg integration lives in :mod:`h5coro_hidefix.zagg_backend` and is only
imported by zagg's entry-point discovery — importing this package alone never
pulls zagg, pandas, or pyarrow.
"""

from h5coro_hidefix._native import Index, __version__

__all__ = ["Index", "__version__"]

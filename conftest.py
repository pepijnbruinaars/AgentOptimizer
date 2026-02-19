"""Pytest configuration for lightweight dependency shims."""

from __future__ import annotations

import sys
import types


def _ensure_tqdm_stub() -> None:
    """Provide a no-op tqdm implementation when tqdm is not installed."""
    try:
        import tqdm  # noqa: F401
    except ModuleNotFoundError:
        tqdm_module = types.ModuleType("tqdm")

        class _NoOpTqdm:
            def __init__(self, *args, **kwargs) -> None:
                pass

            def set_postfix_str(self, *args, **kwargs) -> None:
                pass

            def update(self, *args, **kwargs) -> None:
                pass

            def close(self) -> None:
                pass

        def _tqdm(*args, **kwargs):
            return _NoOpTqdm(*args, **kwargs)

        tqdm_module.tqdm = _tqdm  # type: ignore[attr-defined]
        sys.modules["tqdm"] = tqdm_module


_ensure_tqdm_stub()

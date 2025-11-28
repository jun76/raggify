from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .exts import Exts
    from .metadata import BasicMetaData

__all__ = ["Exts", "BasicMetaData"]


def __getattr__(name: str):
    if name == "Exts":
        from .exts import Exts as _Exts

        return _Exts
    if name == "BasicMetaData":
        from .metadata import BasicMetaData as _BasicMetaData

        return _BasicMetaData

    raise AttributeError(f"module {__name__} has no attribute {name!r}")

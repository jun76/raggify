__all__ = ["FileLoader", "HTMLLoader"]


def __getattr__(name: str):
    if name == "FileLoader":
        from .file_loader import FileLoader

        return FileLoader
    if name == "HTMLLoader":
        from .html_loader import HTMLLoader

        return HTMLLoader
    raise AttributeError(f"module {__name__} has no attribute {name}")

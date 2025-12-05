__all__ = ["FileLoader", "WebPageLoader"]  # type: ignore


def __getattr__(name: str):
    if name == "FileLoader":
        from .file_loader import FileLoader

        return FileLoader
    if name == "WebPageLoader":
        from .web_page_loader import WebPageLoader

        return WebPageLoader
    raise AttributeError(f"module {__name__} has no attribute {name}")

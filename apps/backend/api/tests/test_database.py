from bovi_api.database import _is_azure_files_sqlite


def test_detects_azure_files_sqlite_urls():
    assert _is_azure_files_sqlite("sqlite+aiosqlite:////mnt/data/bovi.db")
    assert _is_azure_files_sqlite(
        "sqlite+aiosqlite:///file:/mnt/data/bovi.db?vfs=unix-dotfile&uri=true"
    )
    assert not _is_azure_files_sqlite("sqlite+aiosqlite:///./bovi.db")

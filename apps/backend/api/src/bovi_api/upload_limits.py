"""Upload size validation helpers."""

from fastapi import HTTPException, UploadFile


def format_file_size(size_bytes: int) -> str:
    """Format byte counts for user-facing API errors."""
    units = ("bytes", "KB", "MB", "GB")
    size = float(size_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            if unit == "bytes":
                return f"{int(size)} bytes"
            if size.is_integer():
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size_bytes} bytes"


def upload_too_large_message(
    *,
    filename: str | None,
    actual_size: int,
    max_size: int,
) -> str:
    """Build a consistent explanation for upload size failures."""
    display_name = filename or "CSV file"
    return (
        f"'{display_name}' is {format_file_size(actual_size)}, which exceeds the "
        f"{format_file_size(max_size)} upload limit. Split the file into smaller CSV files "
        "and upload them separately."
    )


def ensure_upload_file_size(file: UploadFile, *, max_size: int) -> None:
    """Reject uploads whose multipart metadata already reports an oversize file."""
    if file.size is not None and file.size > max_size:
        raise HTTPException(
            status_code=413,
            detail=upload_too_large_message(
                filename=file.filename,
                actual_size=file.size,
                max_size=max_size,
            ),
        )


def ensure_upload_bytes_size(data: bytes, *, filename: str | None, max_size: int) -> None:
    """Reject uploads whose read payload exceeds the configured limit."""
    if len(data) > max_size:
        raise HTTPException(
            status_code=413,
            detail=upload_too_large_message(
                filename=filename,
                actual_size=len(data),
                max_size=max_size,
            ),
        )

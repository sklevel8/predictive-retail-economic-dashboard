"""Domain-level exceptions shared by the service layer."""

from __future__ import annotations


class DomainError(Exception):
    """Base class for errors that map to a 4xx API response."""

    status_code = 400

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class AssetNotFoundError(DomainError):
    status_code = 404

    def __init__(self, asset_name: str) -> None:
        super().__init__(
            f"Asset '{asset_name}' was not found for this tenant. "
            "Upload a dataset for the asset before requesting a forecast."
        )


class InsufficientDataError(DomainError):
    status_code = 422

    def __init__(self, rows: int, required: int) -> None:
        super().__init__(
            f"Not enough history to train a model: {rows} usable rows, "
            f"{required} required."
        )


class UploadValidationError(DomainError):
    status_code = 422

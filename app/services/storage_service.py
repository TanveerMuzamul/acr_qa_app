"""Helpers for optional AWS S3 upload backup.

This module keeps AWS logic isolated from the main Flask routes so the project
stays easier to read and maintain.
"""

from __future__ import annotations

from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from flask import current_app


class StorageServiceError(Exception):
    """Raised when S3 upload fails."""


class StorageService:
    """Simple wrapper around AWS S3 operations used by this project."""

    def __init__(self) -> None:
        self.enabled = bool(current_app.config.get("AWS_S3_ENABLED"))
        self.bucket = current_app.config.get("AWS_S3_BUCKET", "")
        self.prefix = current_app.config.get("AWS_S3_PREFIX", "mri-qa").strip("/")
        self.region = current_app.config.get("AWS_REGION", "")

    def _client(self):
        """Create an S3 client using values from the Flask config."""
        return boto3.client(
            "s3",
            region_name=self.region or None,
            aws_access_key_id=current_app.config.get("AWS_ACCESS_KEY_ID") or None,
            aws_secret_access_key=current_app.config.get("AWS_SECRET_ACCESS_KEY") or None,
        )

    def upload_file(self, file_path: str, object_name: str | None = None) -> str | None:
        """Upload one local file to S3 and return the s3:// path.

        Returns None when AWS S3 integration is disabled.
        """
        if not self.enabled:
            return None
        if not self.bucket:
            raise StorageServiceError("AWS_S3_BUCKET is missing.")

        local_path = Path(file_path)
        key = object_name or f"{self.prefix}/{local_path.name}"
        key = key.lstrip("/")

        try:
            self._client().upload_file(str(local_path), self.bucket, key)
        except (BotoCoreError, ClientError) as exc:
            raise StorageServiceError(f"S3 upload failed: {exc}") from exc

        return f"s3://{self.bucket}/{key}"

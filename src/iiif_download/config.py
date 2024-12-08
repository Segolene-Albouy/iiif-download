"""
Configuration module for iiif_download package.

This module handles all configurable parameters of the package,
providing both default values and methods to override them.
"""

import os
from pathlib import Path
from typing import Optional


class Config:
    """Global configuration for the iiif_download package."""

    def __init__(self):
        self._base_dir = Path(__file__).resolve().parent.parent.parent
        self._img_dir = self._base_dir / "img"
        self._log_dir = self._base_dir / "log"

        # Image processing settings
        self._max_size = 2500
        self._min_size = 1000
        self._max_res = 300
        self._allow_truncation = False

        # Network settings
        self._timeout = 30
        self._retry_attempts = 3
        self._sleep_time = {"default": 0.25, "gallica": 12}

        # Dev settings
        self._debug = False
        self._save_manifest = False

        # Initialize from environment variables if present
        self._load_from_env()

        # Create directories if they don't exist
        self._create_dirs()

    def _load_from_env(self):
        """Load configuration from environment variables."""
        if path := os.getenv("IIIF_BASE_DIR"):
            self._base_dir = Path(path)
            self._img_dir = self._base_dir / "img"
            self._log_dir = self._base_dir / "log"

        if dir_name := os.getenv("IIIF_IMG_DIR"):
            self._img_dir = self._base_dir / dir_name

        if dir_name := os.getenv("IIIF_LOG_DIR"):
            self._log_dir = self._base_dir / dir_name

        if size := os.getenv("IIIF_MAX_SIZE"):
            self._max_size = int(size)

        if size := os.getenv("IIIF_MIN_SIZE"):
            self._min_size = int(size)

        if res := os.getenv("IIIF_MAX_RESOLUTION"):
            self._max_res = int(res)

        if truncation := os.getenv("IIIF_ALLOW_TRUNCATION"):
            self._allow_truncation = truncation.lower() in ("true", "1", "yes")

        if timeout := os.getenv("IIIF_TIMEOUT"):
            self._timeout = int(timeout)

        if retries := os.getenv("IIIF_RETRY_ATTEMPTS"):
            self._retry_attempts = int(retries)

        if sleep_time := os.getenv("IIIF_SLEEP"):
            self._sleep_time = {"default": float(sleep_time), "gallica": 12}

        if debug := os.getenv("IIIF_DEBUG"):
            self._debug = debug.lower() in ("true", "1", "yes")

        if save := os.getenv("IIIF_SAVE_MANIFEST"):
            self._save_manifest = save.lower() in ("true", "1", "yes")

    def _create_dirs(self):
        """Create necessary directories if they don't exist."""
        self._img_dir.mkdir(parents=True, exist_ok=True)
        self._log_dir.mkdir(parents=True, exist_ok=True)

    @property
    def img_dir(self) -> Path:
        """Directory where images will be saved."""
        return self._img_dir

    @img_dir.setter
    def img_dir(self, path):
        if not isinstance(path, (str, Path)):
            raise TypeError("Path must be Path or string")
        self._img_dir = Path(path)
        self._img_dir.mkdir(parents=True, exist_ok=True)

    @property
    def log_dir(self) -> Path:
        """Directory where logs will be saved."""
        return self._log_dir

    @log_dir.setter
    def log_dir(self, path: Path):
        if not isinstance(path, (str, Path)):
            raise TypeError("Path must be Path or string")
        self._log_dir = Path(path)
        self._log_dir.mkdir(parents=True, exist_ok=True)

    @property
    def max_size(self) -> int:
        """Maximum size for image dimensions."""
        return self._max_size

    @max_size.setter
    def max_size(self, value: int):
        if value < 0:
            raise ValueError("max_size must be positive")
        self._max_size = value

    @property
    def min_size(self) -> int:
        """Minimum size for image dimensions."""
        return self._min_size

    @min_size.setter
    def min_size(self, value: int):
        if value < 0:
            raise ValueError("min_size must be positive")
        if value > self.max_size:
            raise ValueError("min_size cannot be larger than max_size")
        self._min_size = value

    @property
    def max_res(self) -> int:
        """Maximum resolution for saved images."""
        # TODO use
        return self._max_res

    @max_res.setter
    def max_res(self, value):
        """Maximum resolution for saved images."""
        if value < 0:
            raise ValueError("max_res must be positive")
        self._max_res = value

    @property
    def timeout(self) -> int:
        """Timeout in seconds for network requests."""
        # TODO use
        return self._timeout

    @timeout.setter
    def timeout(self, value: int):
        if value < 0:
            raise ValueError("Timeout must be positive")
        self._timeout = value

    @property
    def retry_attempts(self) -> int:
        """Number of retry attempts for failed downloads."""
        # TODO use
        return self._retry_attempts

    @retry_attempts.setter
    def retry_attempts(self, value: int):
        if value < 0:
            raise ValueError("Retry attempts must be positive")
        self._retry_attempts = value

    @property
    def sleep_time(self) -> dict:
        """Sleep time between requests for different providers."""
        return self._sleep_time.copy()

    def set_sleep_time(self, provider: str, value: float) -> None:
        """
        Set sleep time for a specific provider.
        """
        if not isinstance(value, (int, float)):
            raise TypeError("Sleep time must be a number")
        if value <= 0:
            raise ValueError("Sleep time must be positive")

        self._sleep_time[provider] = float(value)

    def get_sleep_time(self, url: Optional[str] = None) -> float:
        """Get sleep time for a specific URL."""
        if url and "gallica" in url:
            return self._sleep_time["gallica"]
        return self._sleep_time["default"]

    @property
    def debug(self) -> bool:
        """Enable debug mode."""
        return self._debug

    @debug.setter
    def debug(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError("Debug must be a boolean")
        self._debug = value

    @property
    def save_manifest(self) -> bool:
        """Enable debug mode."""
        return self._save_manifest

    @save_manifest.setter
    def save_manifest(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError("Save manifest must be a boolean")
        self._save_manifest = value

    @property
    def allow_truncation(self) -> bool:
        """Allow truncation of images."""
        return self._allow_truncation

    @allow_truncation.setter
    def allow_truncation(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError("Allow truncation must be a boolean")
        self._allow_truncation = value


# Global configuration instance
config = Config()

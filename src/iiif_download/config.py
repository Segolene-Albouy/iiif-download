"""
Configuration module for iiif_download package.

This module handles all configurable parameters of the package,
providing both default values and methods to override them.
"""

import asyncio
import copy
import os
from asyncio import Lock, Semaphore
from pathlib import Path
from time import time
from typing import Dict, Optional, Union
from urllib.parse import urlparse


class Config:
    """Global configuration for the iiif_download package."""

    def __init__(self, **kwargs):
        # current absolute dir where the user is executing python
        self._base_dir = Path(os.getcwd()).resolve()
        self._img_dir = self._base_dir / "img"
        self._log_dir = self._base_dir / "log"

        # Image processing settings
        self._max_size = 2500
        self._min_size = 1000
        self._max_res = 300
        self._allow_truncation = False

        # Network settings
        self._retry_attempts = 3
        self._sleep_time = {"default": 0.05, "gallica.bnf.fr": 12}
        self._domain_locks = {}  # {domain: {'lock': Lock(), 'last_time': float}}
        self._threads = 20
        self._semaphore = Semaphore(self._threads)
        self._timeout = 120  # 2 minutes
        self._user_agent = "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:102.0) Gecko/20100101 Firefox/102.0"
        self._proxy_settings = {}

        # Dev settings
        self._debug = False
        self._is_logged = True
        self._save_manifest = False

        for key, value in kwargs.items():
            # override any config attribute
            setattr(self, f"_{key}", value)

        # Initialize from environment variables if present
        self._load_from_env()

        # Create directories if they don't exist
        self._create_dirs()

    def _load_from_env(self):
        """Load configuration from environment variables."""
        if path := os.getenv("IIIF_BASE_DIR"):
            self._base_dir = Path(path)

        if dir_name := os.getenv("IIIF_IMG_DIR"):
            self.img_dir = dir_name

        if dir_name := os.getenv("IIIF_LOG_DIR"):
            self.log_dir = dir_name

        if size := os.getenv("IIIF_MAX_SIZE"):
            self._max_size = int(size)

        if size := os.getenv("IIIF_MIN_SIZE"):
            self._min_size = int(size)

        if res := os.getenv("IIIF_MAX_RESOLUTION"):
            self._max_res = int(res)

        if truncation := os.getenv("IIIF_ALLOW_TRUNCATION"):
            self._allow_truncation = truncation.lower() in ("true", "1", "yes")

        if retries := os.getenv("IIIF_RETRY_ATTEMPTS"):
            self._retry_attempts = int(retries)

        if sleep_time := os.getenv("IIIF_SLEEP"):
            self._sleep_time = {"default": float(sleep_time), "gallica": 12}

        if threads := os.getenv("IIIF_THREADS"):
            self._threads = int(threads)
            self._semaphore = Semaphore(self._threads)

        if debug := os.getenv("IIIF_DEBUG"):
            self._debug = debug.lower() in ("true", "1", "yes")

        if save := os.getenv("IIIF_SAVE_MANIFEST"):
            self._save_manifest = save.lower() in ("true", "1", "yes")

        if http_proxy := os.environ.get("http_proxy") or os.environ.get("HTTP_PROXY"):
            self._proxy_settings["http"] = http_proxy

        if https_proxy := os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY"):
            self._proxy_settings["https"] = https_proxy

        # TODO add is_logged, semaphore, user_agent, save_manifest

    def _create_dirs(self):
        """Create necessary directories if they don't exist."""
        self._img_dir.mkdir(parents=True, exist_ok=True)
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def set_path(
        self, path: Optional[Union[str, Path]] = None, base_dir: Optional[Union[str, Path]] = None
    ) -> Path:
        parent_dir = Path(base_dir).resolve() if base_dir else self.base_dir
        if not path:
            return parent_dir
        if not isinstance(path, (str, Path)):
            raise TypeError("path must be Path or string")

        path = Path(path)
        child_dir = path if path.is_absolute() else parent_dir / path
        child_dir.mkdir(parents=True, exist_ok=True)

        return child_dir

    @property
    def base_dir(self) -> Path:
        """Base directory for logs and images"""
        return self._base_dir.resolve()

    @base_dir.setter
    def base_dir(self, path):
        # if path is absolute, base_dir will not be taken into account
        self._base_dir = self.set_path(path, base_dir=Path(os.getcwd()).resolve())

    @property
    def img_dir(self) -> Path:
        """Directory where images will be saved."""
        return self._img_dir

    @img_dir.setter
    def img_dir(self, path):
        self._img_dir = self.set_path(path)

    @property
    def log_dir(self) -> Path:
        """Directory where logs will be saved."""
        return self._log_dir

    @log_dir.setter
    def log_dir(self, path: Path):
        self._log_dir = self.set_path(path)

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

    def set_sleep_time(self, sec_nb: float, provider_domain: str = "default") -> None:
        """
        Set sleep time for a specific provider.
        """
        if not isinstance(sec_nb, (int, float)):
            raise TypeError("Sleep time must be a number")
        if sec_nb <= 0:
            raise ValueError("Sleep time must be positive")
        if not isinstance(provider_domain, str):
            raise TypeError("Provider domain must be a string")

        if provider_domain != "default":
            parsed = urlparse(provider_domain if "://" in provider_domain else f"http://{provider_domain}")
            provider_domain = parsed.netloc or "default"

        self._sleep_time[provider_domain] = float(sec_nb)

    def get_sleep_time(self, url: Optional[str] = None) -> float:
        """Get sleep time for a specific URL."""
        if url:
            domain = urlparse(url).netloc
            if domain in self._sleep_time:
                return self._sleep_time[domain]
        return self._sleep_time["default"]

    def get_domain_lock(self, url: str):
        domain = urlparse(url).netloc
        if domain not in self._domain_locks:
            self._domain_locks[domain] = {"lock": Lock(), "last_time": 0}
        return self._domain_locks[domain]

    async def wait_for_domain(self, url: str):
        """Wait appropriate time before making request to respect rate limits per domain."""
        sleep_time = self.get_sleep_time(url)
        lock_info = self.get_domain_lock(url)

        async with lock_info["lock"]:
            elapsed = time() - lock_info["last_time"]
            if elapsed < sleep_time:
                await asyncio.sleep(sleep_time - elapsed)
            lock_info["last_time"] = time()

    @property
    def threads(self) -> int:
        """Number of concurrent threads for downloads."""
        return self._threads

    @threads.setter
    def threads(self, value: int):
        if value < 1:
            raise ValueError("Thread count must be at least 1")
        self._threads = value
        self._semaphore = Semaphore(value)

    @property
    def semaphore(self) -> Semaphore:
        return self._semaphore

    @semaphore.setter
    def semaphore(self, value: int):
        if value < 0:
            raise ValueError("Semaphore value must be positive")
        self._semaphore = Semaphore(value)

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
    def user_agent(self) -> str:
        return self._user_agent

    @user_agent.setter
    def user_agent(self, value: str):
        if not isinstance(value, str):
            raise TypeError("User agent must be a string 'Browser/V. (OS) Platform/V.'")
        self._user_agent = value

    @property
    def is_logged(self) -> bool:
        """Save logs to file."""
        return self._is_logged

    @is_logged.setter
    def is_logged(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError("is_logged must be a boolean")
        self._is_logged = value

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

    @property
    def proxy_settings(self) -> Dict[str, Optional[str]]:
        return self._proxy_settings.copy()

    @proxy_settings.setter
    def proxy_settings(self, settings: Dict[str, str]):
        if not isinstance(settings, dict):
            raise TypeError("Proxy settings must be a dictionary")
        self._proxy_settings = settings

    @property
    def timeout(self) -> int:
        return self._timeout

    @timeout.setter
    def timeout(self, value: int):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("Timeout must be a positive integer")
        self._timeout = value

    def copy(self):
        return copy.copy(self)


# Global configuration instance
config = Config()

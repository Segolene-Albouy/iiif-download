import time
import json
import logging
import traceback
from pathlib import Path
from typing import Any, Optional, Union
from contextlib import contextmanager
from functools import wraps

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn
)

from ..config import config


def sanitize(v):
    """
    Helper function to convert non-serializable values to string representations.
    """
    if isinstance(v, (str, int, float, bool, type(None))):
        return v
    elif isinstance(v, (list, tuple)):
        return [sanitize(x) for x in v]
    elif isinstance(v, dict):
        return {str(k): sanitize(val) for k, val in v.items()}
    else:
        # For custom objects, include class name in representation
        return f"{v.__class__.__name__}({str(v)})"


def pprint(o):
    if isinstance(o, str):
        if "html" in o:
            from . import strip_tags

            return strip_tags(o)[:500]
        try:
            return json.dumps(json.loads(o), indent=4, sort_keys=True)
        except ValueError:
            return o
    elif isinstance(o, dict) or isinstance(o, list):
        try:
            return json.dumps(o, indent=4, sort_keys=True)
        except TypeError:
            try:
                if isinstance(o, dict):
                    sanitized = {str(k): sanitize(v) for k, v in o.items()}
                else:
                    sanitized = [sanitize(v) for v in o]
                return json.dumps(sanitized, indent=4, sort_keys=True)
            except Exception:
                return str(o)
    return str(o)


class Logger:
    def __init__(self, log_dir: Union[str, Path]):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.error_log = self.log_dir / "error.log"
        self.download_log = self.log_dir / "download_fails.log"

        self.console = Console()
        self._quiet = False

        if config.is_logged:
            self.file_logger = logging.getLogger("iiif-downloader")
            self.file_logger.setLevel(logging.ERROR)
            self.file_logger.propagate = False

            fh = logging.FileHandler(self.error_log)
            fh.setLevel(logging.ERROR)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            self.file_logger.addHandler(fh)

    @contextmanager
    def quiet_mode(self):
        """Suppress console output during progress operations."""
        old_quiet = self._quiet
        self._quiet = True
        try:
            yield
        finally:
            self._quiet = old_quiet

    def error(self, *msg: Any, exception: Optional[Exception] = None):
        """Log error message."""
        message = self.format(*msg)

        if config.is_logged:
            error_msg = message
            if exception:
                error_msg += f"\n{traceback.format_exc()}"
            self.file_logger.error(error_msg)

        if not self._quiet:
            self.console.print(f"[red]❌ {message}[/red]")

    def warning(self, *msg: Any):
        """Log warning message."""
        message = self.format(*msg)
        if not self._quiet:
            self.console.print(f"[yellow]⚠️  {message}[/yellow]")

    def info(self, *msg: Any):
        """Log info message."""
        message = self.format(*msg)
        if not self._quiet:
            self.console.print(f"[blue]ℹ️  {message}[/blue]")

    def success(self, *msg: Any):
        """Log success message."""
        message = self.format(*msg)
        if not self._quiet:
            self.console.print(f"[green]✅ {message}[/green]")

    def log(self, *msg: Any, msg_type: Optional[str] = None):
        """Log a message with a given type."""
        msg_type = msg_type or "info"
        if msg_type == "error":
            self.error(*msg)
        elif msg_type == "warning":
            self.warning(*msg)
        elif msg_type == "success":
            self.success(*msg)
        else:
            self.info(*msg)

    def create_progress(self, description: str = "Processing") -> Progress:
        """Create a Rich progress bar with standard columns."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=self.console,
            transient=False
        )

    def log_failed_download(self, img_path: str, img_url: str):
        """Log a failed download attempt."""
        with open(self.download_log, "a") as f:
            f.write(f"{img_path} {img_url}\n")

    def log_failed_manifests(self, manifest_url: str):
        """Log a failed manifest download attempt."""
        with open(self.download_log, "a") as f:
            f.write(f"{manifest_url}\n")

    @staticmethod
    def add_to_json(log_file: Path, content: dict, mode: str = "w"):
        """Add content to JSON log file."""
        with open(log_file, mode) as f:
            json.dump(content, f, indent=2)

    @staticmethod
    def format(*msg: Any) -> str:
        """Format message for logging."""
        return "\n\n".join(pprint(m) for m in msg)

    @staticmethod
    def format_exception(exception: Exception) -> str:
        """Format exception for logging."""
        return f"[{exception.__class__.__name__}] {exception}\n{traceback.format_exc()}"


# Create a global logger instance
logger = Logger(config.log_dir)


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time

        if execution_time < 0.1:
            msg_type = "success"
        elif execution_time < 0.5:
            msg_type = "info"
        elif execution_time < 1:
            msg_type = "warning"
        else:
            msg_type = "error"

        logger.log(f"\n[{func.__name__}]: {execution_time:.3f} secondes", msg_type=msg_type)
        return result

    return wrapper

"""
IIIF Downloader
==============

A Python package to download images from IIIF manifests.
"""

from .config import Config, config
from .image import IIIFImage
from .manifest import IIIFManifest

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.0.0+unknown"

__all__ = ["IIIFManifest", "IIIFImage", "config", "Config"]

import asyncio
import gc
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import unquote

from .config import Config, config
from .image import IIIFImage
from .utils import (
    create_dir,
    get_id,
    get_json_async,
    get_license_url,
    get_meta_value,
    mono_val,
    sanitize_str,
)
from .utils.logger import logger

LICENSE = [
    "license",
    "licence",
    "lizenz",
    "rights",
    "droits",
    "access",
    "copyright",
    "rechteinformationen",
    "conditions",
]


class IIIFManifest:
    """Represents a IIIF manifest with its metadata and image list."""

    def __init__(
        self,
        url: str,
        save_dir: Optional[Union[Path, str]] = None,
        prefix: str = "",
        leading_zeros: int = 4,
        conf: Config = config,
        **kwargs,
    ):
        self.config = conf

        if kwargs:
            self.config = conf.copy()
            for key, value in kwargs.items():
                # override any config value
                setattr(self.config, key, value)

        self.url = unquote(url)
        self.content: Optional[Dict[str, Any]] = None
        self._prefix = prefix
        self._leading_zeros = leading_zeros
        self._save_dir: Path = self.config.set_path(save_dir, self.config.img_dir)
        self._manifest_info: Dict = {}
        self._license: Optional[str] = None
        self._resources: Optional[List] = None
        self._images: Optional[List[IIIFImage]] = None

    @property
    def save_dir(self) -> Path:
        """Directory where images will be saved."""
        return self._save_dir

    @save_dir.setter
    def save_dir(self, path):
        self._save_dir = self.config.set_path(path, self.config.img_dir)

    @property
    def uid(self) -> str:
        """Generate a directory name from manifest URL."""
        return sanitize_str(self.url).replace("manifest", "").replace("json", "")

    async def load(self, reload=False) -> bool:
        """Load manifest content from URL."""
        if bool(self.content) and not reload:
            return True

        try:
            self.content = await get_json_async(self.url, allow_insecure=True)
            if self.config.save_manifest:
                with open(self.save_dir / "manifest.json", "w") as f:
                    json.dump(self.content, f)
            return bool(self.content)
        except Exception as e:
            logger.error(f"Failed to load manifest from {self.url}", exception=e)
            logger.log_failed_manifests(self.url)
            return False

    def get_meta(self, label: str) -> Optional[str]:
        if not self.content or "metadata" not in self.content:
            return None

        for meta in self.content.get("metadata", []):
            if value := get_meta_value(meta, label):
                return value
        return None

    # TODO add a property metadata with every metadata provided
    # TODO add a property canvas

    @property
    def license(self) -> str:
        if self._license is None:
            self._license = self.get_license()
        return self._license

    def get_license(self) -> str:
        """Get license information from manifest."""
        if not self.content:
            return "No manifest loaded"

        for label in ["license", "rights"]:
            if lic := self.content.get(label):
                return get_license_url(mono_val(lic))

        if metadata := self.content.get("metadata"):
            for meta in metadata:
                if meta_label := str(meta.get("label", "")).lower():
                    if any(term in meta_label for term in LICENSE):
                        return get_license_url(meta.get("value", ""))

                for label in LICENSE:
                    if value := get_meta_value(meta, label):
                        return get_license_url(value)

        return get_license_url(mono_val(self.content.get("attribution", "")))

    @staticmethod
    def get_image_resource(image_data: Dict[str, Any], label: str = "") -> Optional[Dict[str, Any]]:
        """Extract image resource from image data."""
        try:
            resource = image_data.get("resource") or image_data.get("body")
            if label:
                resource["label"] = label
            return resource
        except (KeyError, TypeError):
            return None

    @property
    def resources(self) -> List:
        if self._resources is None:
            self._resources = self.get_resources()
        return self._resources

    def get_resources(self) -> List:
        """Extract all image resources from manifest."""
        resources = []
        if not self.content:
            return resources

        try:
            # Try sequences/canvases path
            sequences = self.content["sequences"]
            if len(sequences) < 1:
                return resources
            canvases = self.content["sequences"][0]["canvases"]
            for canvas in canvases:
                label = canvas.get("label", "")
                for image in canvas["images"]:
                    if resource := self.get_image_resource(image, label=label):
                        resources.append(resource)
        except KeyError:
            try:
                # Try items path
                items = self.content["items"]
                for item in items:
                    for sub_item in item["items"][0]["items"]:
                        if resource := self.get_image_resource(sub_item):
                            resources.append(resource)
            except (KeyError, IndexError, TypeError) as e:
                logger.error("Failed to extract images from manifest", exception=e)

        return resources

    @staticmethod
    def get_img_service(resource):
        if resource.get("service"):
            return get_id(resource["service"])
        img_id = get_id(resource)

        # look for hidden image services
        if img_id.endswith(("/full/full/0/default.jpg", "/full/max/0/default.jpg")):
            return img_id.rsplit("/", 4)[0]

        # case were only static images are provided
        return img_id

    @property
    def images(self) -> List:
        if self._images is None:
            self._images = self.get_images()
        return self._images

    def get_images(self) -> List[IIIFImage]:
        images = []
        for i, resource in enumerate(self.get_resources()):
            images.append(
                IIIFImage(
                    idx=i + 1,
                    img_id=self.get_img_service(resource),
                    resource=resource,
                    save_dir=self.save_dir,
                    prefix=self._prefix,
                    leading_zeros=self._leading_zeros,
                )
            )
        return images

    def save_log(self):
        if self.config.is_logged:
            logger.add_to_json(self.save_dir / "info.json", self._manifest_info)

    async def _async_download_manifest(self, show_progress: bool = False):
        import time
        start_time = time.perf_counter()

        if self.config.is_logged:
            self._manifest_info = {"url": self.url, "license": "", "images": {}}

        if not await self.load():
            logger.warning(f"Unable to load json content of {self.url}")
            self.save_log()
            return self

        if self.config.is_logged:
            self._manifest_info["license"] = self.license

        images = self.images
        if not images:
            logger.warning(f"No images found in manifest {self.url}")
            self.save_log()
            return self

        logger.info(f"Downloading {len(images)} images from {self.url}")

        if self.config.debug:
            images = images[:6]

        # Semaphore local (event-loop safe)
        semaphore = asyncio.Semaphore(self.config.threads)

        async def download_image(image, progress_updater=None):
            async with semaphore:
                result = await image.save()
                if self.config.is_logged and result:
                    self._manifest_info["images"][image.img_name] = image.sized_url()
                if progress_updater:
                    progress_updater()
                return result

        progress = logger.create_progress() if show_progress else nullcontext()
        with progress:
            task_id = progress.add_task(f"[green]Downloading {len(images)} image(s)", total=len(images)) if show_progress else None
            updater = lambda: progress.update(task_id, advance=1) if show_progress else None

            results = await asyncio.gather(
                *[download_image(img, updater) for img in images],
                return_exceptions=True
            )

        success_count = sum(1 for r in results if r is True)
        failed_count = len(results) - success_count
        elapsed = time.perf_counter() - start_time

        if show_progress:
            logger.success(
                f"Downloaded {success_count}/{len(images)} image(s) in {elapsed:.1f}s ({success_count / elapsed:.1f} img/s)"
            )
            if failed_count > 0:
                logger.warning(f"{failed_count}/{len(images)} image(s) failed")

        self.save_log()
        await self.config.close_session()
        return self

    def download(
        self, save_dir: Optional[Union[Path, str]] = None, cleanup: bool = False, show_progress: bool = False
    ) -> "IIIFManifest":
        if save_dir:
            self.save_dir = save_dir
        if not self.save_dir.exists():
            create_dir(self.save_dir)

        try:
            result = asyncio.run(self._async_download_manifest(show_progress=show_progress))
        finally:
            if cleanup:
                self.cleanup()
        return result

    def cleanup(self):
        self.content = None
        self._resources = None
        self._manifest_info = None
        if self._images:
            for img in self._images:
                img.cleanup()
            self._images = None
        gc.collect()

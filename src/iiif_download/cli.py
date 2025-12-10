#!/usr/bin/env python3

import argparse
from pathlib import Path

from .config import config
from .manifest import IIIFManifest
from .utils.logger import logger


def main():
    parser = argparse.ArgumentParser(
        description="Download IIIF manifest images",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("manifest_url", type=str, nargs="?", help="Single manifest URL to download")
    parser.add_argument("-f", "--file", type=str, help="File containing manifest URLs, one per line")
    parser.add_argument("-d", "--img_dir", type=str, help="Path where to save downloaded images")
    parser.add_argument("-t", "--threads", type=int, default=20, help="Number of concurrent threads (default: 20)")
    parser.add_argument("-q", "--quiet", action="store_true", help="Minimal output (progress bars only)")
    args = parser.parse_args()

    if not args.manifest_url and not args.file:
        logger.error("You must provide either a manifest URL or a file with URLs")
        parser.print_usage()
        return 1

    manifests = []
    if args.file:
        try:
            with open(args.file) as f:
                manifests.extend(
                    line.strip() for line in f
                    if line.strip() and not line.strip().startswith("#")
                )
        except Exception as e:
            logger.error(f"Failed to read file {args.file}", exception=e)
            return 1

    if args.manifest_url:
        manifests.append(args.manifest_url)

    if not manifests:
        logger.error("No valid manifest URLs found")
        return 1

    only_one = len(manifests) == 1
    config.img_dir = args.img_dir or config.img_dir
    config.threads = args.threads

    if not args.quiet:
        logger.info(f"Starting download: {len(manifests)} manifest(s), {config.threads} threads")
        logger.info(f"Output directory: {config.img_dir}")
        logger.console.print()

    progress = logger.create_progress()

    with progress:
        manifest_task = progress.add_task(
            f"[cyan]Processing {len(manifests)} manifest(s)",
            total=len(manifests)
        )

        success_count = 0
        failed_count = 0

        for url in manifests:
            progress.update(manifest_task, description=f"[cyan]Manifest: {url}")

            try:
                manifest = IIIFManifest(url)
                manifest.save_dir = None if only_one else manifest.uid

                with logger.quiet_mode():
                    result = manifest.download(cleanup=not only_one, show_progress=True)

                if result:
                    success_count += 1
                else:
                    failed_count += 1

            except Exception as e:
                logger.error(f"Failed to process {url}", exception=e)
                failed_count += 1

            progress.update(manifest_task, advance=1)

    logger.console.print()
    if success_count > 0:
        logger.success(f"Downloaded {success_count} manifest(s) successfully")
    if failed_count > 0:
        logger.warning(f"{failed_count} manifest(s) failed")

    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    exit(main())

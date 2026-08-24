# Modified from: https://github.com/ayushjain1144/odin/tree/0cd49cb3a52e88869e0a983a1b2f2d6277041b9e/data_preparation
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Customized script to download only the necessary ScanNet files.

You still need to get the original script from the authors of ScanNet.

This assumes that you have the download-scannet.py file in this subfolder.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_DOWNLOADER = Path(__file__).with_name("download-scannet-v2.py")
CACHE_DIR = Path(__file__).resolve().parent / ".cache"
REQUIRED_FILETYPES = [
    '.aggregation.json',
    '.txt',
    '_vh_clean_2.0.010000.segs.json',
    '_vh_clean_2.ply',
    '_vh_clean_2.labels.ply',
]


def get_scan_ids():
    """Load the .csv files and return a set of scan_ids."""
    scan_ids = []
    with (ROOT / 'splits/scannet_splits/scannetv2_trainval.txt').open() as fid:
        scan_ids += fid.readlines()
    return sorted(list(set(sid.strip('\n') for sid in scan_ids)))


def is_scan_complete(scan_id, out_dir: Path):
    """Return True if all required files already exist for a scan."""
    scan_dir = out_dir / "scans" / scan_id
    for filetype in REQUIRED_FILETYPES:
        if not (scan_dir / f"{scan_id}{filetype}").exists():
            return False
    return True


def download_file(scan_id, filetype, out_dir: Path, attempts: int, wait_seconds: int):
    for attempt in range(1, attempts + 1):
        try:
            subprocess.run([
                sys.executable, str(OFFICIAL_DOWNLOADER),
                '-o', str(out_dir),
                '--id', scan_id,
                '--type', filetype,
                '--skip_existing',
                '--cache_dir', str(out_dir / '.cache'),
                '--attempts', str(attempts),
                '--retry_wait', str(wait_seconds),
            ], input='\n', text=True, check=True)
            return
        except subprocess.CalledProcessError:
            if attempt == attempts:
                raise
            print(
                f'[{scan_id} {filetype}] failed; waiting {wait_seconds}s before retry '
                f'({attempt}/{attempts})',
                flush=True,
            )
            time.sleep(wait_seconds)


def download_scan_id(scan_id, out_dir: Path, attempts: int, wait_seconds: int):
    """Download files for a specifed scan_id."""
    for filetype in REQUIRED_FILETYPES:
        download_file(scan_id, filetype, out_dir, attempts, wait_seconds)


def main():
    """Download all necessary files for all scan_ids."""
    parser = argparse.ArgumentParser(
        description='Download the ScanNet v2 files required by Qwen-3D.')
    parser.add_argument(
        '--output', type=Path, default=ROOT / 'data/raw/scannet',
        help='ScanNet output root (default: data/raw/scannet).')
    parser.add_argument(
        '--cache-dir', type=Path, default=CACHE_DIR,
        help='Cache directory for download lists (default: data_preparation/scannet/.cache).')
    parser.add_argument(
        '--full', action='store_true',
        help='Disable resume mode and reattempt all scan ids.')
    parser.add_argument('--attempts', type=int, default=3)
    parser.add_argument('--wait-seconds', type=int, default=120)
    args = parser.parse_args()
    if args.attempts < 1:
        parser.error('--attempts must be at least 1')
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    scan_ids = get_scan_ids()

    if not args.full:
        target_scan_ids = []
        skipped = 0
        for scan_id in scan_ids:
            if is_scan_complete(scan_id, args.output):
                skipped += 1
            else:
                target_scan_ids.append(scan_id)
        print(f'Found {len(target_scan_ids)} missing scans, skipped {skipped} complete scans (resume mode).')
    else:
        target_scan_ids = scan_ids
        print(f'Full mode enabled: {len(target_scan_ids)} scans to process.')

    for scan_id in tqdm(target_scan_ids):
        download_scan_id(scan_id, args.output, args.attempts, args.wait_seconds)


if __name__ == "__main__":
    main()

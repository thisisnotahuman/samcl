#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
import shutil
import urllib.request
import urllib.error
import zipfile
from pathlib import Path

from tqdm import tqdm


# Default to HTTP because some clusters/proxies break SSL hostname verification.
# You can override via --base_url.
COCO_BASE = "http://images.cocodataset.org"


def _open_with_retry(
    url: str,
    *,
    headers: dict[str, str],
    retries: int,
    backoff_seconds: float,
    timeout: float,
) -> urllib.request.addinfourl:
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            last_err = e
            # COCO mirror may throttle: 503 Slow Down / 429 Too Many Requests
            if e.code in (429, 503):
                sleep_s = backoff_seconds * (2**attempt)
                print(f"[warn] {e.code} {e.reason}; sleep {sleep_s:.1f}s then retry ({attempt+1}/{retries})")
                time.sleep(sleep_s)
                continue
            raise
        except Exception as e:
            last_err = e
            sleep_s = backoff_seconds * (2**attempt)
            print(f"[warn] download error: {type(e).__name__}: {e}; sleep {sleep_s:.1f}s then retry ({attempt+1}/{retries})")
            time.sleep(sleep_s)
            continue
    assert last_err is not None
    raise last_err


def _download(
    url: str,
    dst: Path,
    *,
    force: bool = False,
    resume: bool = True,
    use_wget: bool = False,
    retries: int = 8,
    backoff_seconds: float = 5.0,
    timeout: float = 60.0,
    user_agent: str = "samcl-coco-downloader/0.1 (+https://images.cocodataset.org)",
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not force and dst.stat().st_size > 0:
        return

    tmp = dst.with_suffix(dst.suffix + ".tmp")
    if force and tmp.exists():
        tmp.unlink()

    # Prefer wget if available (better retry + resume behavior on flaky links).
    if use_wget:
        wget = shutil.which("wget")
        if wget is None:
            print("[warn] --use_wget specified but wget not found; falling back to urllib")
        else:
            import subprocess

            tmp.parent.mkdir(parents=True, exist_ok=True)

            # Detect whether this wget supports --retry-on-http-error (GNU Wget >= ~1.20).
            retry_on_http_error_supported = False
            try:
                help_out = subprocess.run(
                    [wget, "--help"],
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                ).stdout
                retry_on_http_error_supported = "--retry-on-http-error" in help_out
            except Exception:
                retry_on_http_error_supported = False

            # We handle retries/backoff ourselves so that HTTP 503 doesn't crash the script.
            base_cmd = [
                wget,
                "-c",  # continue / resume
                "--tries",
                "1",
                "--read-timeout",
                str(int(timeout)),
                "--timeout",
                str(int(timeout)),
                "--user-agent",
                user_agent,
                "-O",
                str(tmp),
            ]
            if retry_on_http_error_supported:
                # Ask wget to treat 429/503 as retryable within a single attempt too.
                base_cmd += ["--retry-on-http-error=429,503"]

            for attempt in range(retries + 1):
                cmd = base_cmd + [url]
                print("[info] wget:", " ".join(cmd))
                proc = subprocess.run(cmd, check=False)
                if proc.returncode == 0:
                    tmp.replace(dst)
                    return

                sleep_s = backoff_seconds * (2**attempt)
                print(
                    f"[warn] wget failed (exit={proc.returncode}); "
                    f"sleep {sleep_s:.1f}s then retry ({attempt+1}/{retries})"
                )
                time.sleep(sleep_s)

            raise RuntimeError(f"wget download failed after retries: {url}")

    existing = tmp.stat().st_size if (resume and tmp.exists()) else 0
    mode = "ab" if existing > 0 else "wb"
    headers = {"User-Agent": user_agent}
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"

    with _open_with_retry(
        url,
        headers=headers,
        retries=retries,
        backoff_seconds=backoff_seconds,
        timeout=timeout,
    ) as resp:
        # If Range was requested and accepted, status is 206 Partial Content.
        # Content-Length in that case is remaining bytes; we add existing for progress total.
        total = resp.headers.get("Content-Length")
        total_size = int(total) if total is not None else None
        if total_size is not None:
            total_size = total_size + existing

        with tmp.open(mode) as f, tqdm(
            total=total_size,
            initial=existing,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=f"download {dst.name}",
        ) as pbar:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                pbar.update(len(chunk))

    tmp.replace(dst)


def _extract_zip(zip_path: Path, dst_dir: Path, *, force: bool = False) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    marker = dst_dir / ".extracted"
    if marker.exists() and not force:
        return
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dst_dir)
    marker.write_text("ok\n", encoding="utf-8")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_data_dir() -> Path:
    # 用户要求：数据放到 ../data（相对仓库根目录）
    return repo_root().parent / "data"


def main() -> None:
    p = argparse.ArgumentParser("prepare_coco2017")
    p.add_argument("--data_dir", type=str, default=None, help="default: ../data (sibling of repo)")
    p.add_argument("--base_url", type=str, default=COCO_BASE, help="default: http://images.cocodataset.org")
    p.add_argument("--download_train_images", action="store_true")
    p.add_argument("--download_val_images", action="store_true")
    p.add_argument("--download_annotations", action="store_true")
    p.add_argument("--all", action="store_true", help="download train2017+val2017 images and annotations")
    p.add_argument("--force", action="store_true", help="re-download / re-extract")
    p.add_argument("--no_resume", action="store_true", help="disable resume from .tmp files")
    p.add_argument("--use_wget", action="store_true", help="use wget if available (recommended on clusters)")
    p.add_argument("--retries", type=int, default=8)
    p.add_argument("--backoff", type=float, default=5.0, help="initial backoff seconds for 429/503")
    args = p.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir()
    coco_dir = data_dir / "coco2017"
    downloads_dir = coco_dir / "downloads"
    images_dir = coco_dir / "images"
    ann_dir = coco_dir / "annotations"

    if args.all:
        args.download_train_images = True
        args.download_val_images = True
        args.download_annotations = True

    if not (args.download_train_images or args.download_val_images or args.download_annotations):
        # 默认行为：下载全部
        args.download_train_images = True
        args.download_val_images = True
        args.download_annotations = True

    print(f"[info] repo_root: {repo_root()}")
    print(f"[info] data_dir:  {data_dir}")
    print(f"[info] coco_dir:  {coco_dir}")
    base_url = str(args.base_url).rstrip("/")
    print(f"[info] base_url:  {base_url}")
    coco_dir.mkdir(parents=True, exist_ok=True)

    # 1) Images
    if args.download_train_images:
        url = f"{base_url}/zips/train2017.zip"
        zip_path = downloads_dir / "train2017.zip"
        _download(
            url,
            zip_path,
            force=args.force,
            resume=not args.no_resume,
            use_wget=bool(args.use_wget),
            retries=args.retries,
            backoff_seconds=args.backoff,
        )
        _extract_zip(zip_path, images_dir, force=args.force)
        print(f"[ok] train images at: {images_dir / 'train2017'}")

    if args.download_val_images:
        url = f"{base_url}/zips/val2017.zip"
        zip_path = downloads_dir / "val2017.zip"
        _download(
            url,
            zip_path,
            force=args.force,
            resume=not args.no_resume,
            use_wget=bool(args.use_wget),
            retries=args.retries,
            backoff_seconds=args.backoff,
        )
        _extract_zip(zip_path, images_dir, force=args.force)
        print(f"[ok] val images at: {images_dir / 'val2017'}")

    # 2) Annotations (includes captions_train2017.json / captions_val2017.json)
    if args.download_annotations:
        url = f"{base_url}/annotations/annotations_trainval2017.zip"
        zip_path = downloads_dir / "annotations_trainval2017.zip"
        _download(
            url,
            zip_path,
            force=args.force,
            resume=not args.no_resume,
            use_wget=bool(args.use_wget),
            retries=args.retries,
            backoff_seconds=args.backoff,
        )
        _extract_zip(zip_path, coco_dir, force=args.force)

        # After extraction, COCO puts annotations under coco_dir/annotations
        if not ann_dir.exists():
            print("[warn] expected annotations dir not found after extraction.")
        else:
            print(f"[ok] annotations at: {ann_dir}")
            print(f"[ok] captions_train2017.json: {ann_dir / 'captions_train2017.json'}")
            print(f"[ok] captions_val2017.json:   {ann_dir / 'captions_val2017.json'}")

    print("\n[next] 训练时参数示例：")
    print(
        "python -m samcl.train "
        f"--coco_images_dir {images_dir / 'train2017'} "
        f"--coco_captions_json {ann_dir / 'captions_train2017.json'} "
        "--sampling_strategy random"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[abort] interrupted", file=sys.stderr)
        sys.exit(130)

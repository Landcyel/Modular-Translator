"""Vendor directory copy and heavy-weight linking (junction) management.

Layout (vendor root = vendor/ under this file's directory)::

    vendor/
    ├── tools/                  # copied from GPT-SoVITS-main/tools (i18n/audio_sr, etc.)
    └── GPT_SoVITS/             # copied from GPT-SoVITS-main/GPT_SoVITS (inference-only)
        ├── AR/ module/ text/ TTS_infer_pack/ BigVGAN/ feature_extractor/
        ├── f5_tts/ eres2net/ configs/
        ├── sv.py process_ckpt.py
        ├── text/G2PWModel/                        → junction → models/gsv/g2pw/G2PWModel
        └── pretrained_models/
            ├── sv/                                → junction → models/gsv/sv
            ├── fast_langdetect/                   → junction → models/gsv/fast_langdetect
            └── gsv-v4-pretrained/                 → junction → models/v4/gsv-v4-pretrained

Principles:
- Vendored source is byte-identical to upstream (zero modification); imports resolve via sys.path + CWD=vendor root
  (upstream also relies on CWD, see the top of GPT-SoVITS-main/GPT_SoVITS/TTS_infer_pack/TTS.py)
- Heavy weights do not duplicate on disk: use Windows junctions (mklink /J, no admin) pointing into models/;
  if junction creation fails, fall back to a physical copy (copytree)
- This module is idempotent and can be called repeatedly
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENDOR_ROOT = Path(__file__).resolve().parent / "vendor"
PACKAGE_DIR = VENDOR_ROOT / "GPT_SoVITS"
TOOLS_DIR = VENDOR_ROOT / "tools"

SOURCE_ROOT = Path(
    os.environ.get("GSV_SOURCE_ROOT", PROJECT_ROOT / "GPT-SoVITS-main")
)

# Do not copy the GPT_SoVITS/ top-level (training/old GUI/export, not needed for inference)
SKIP_TOP_FILES = {
    "inference_webui.py",
    "inference_webui_fast.py",
    "inference_gui.py",
    "inference_cli.py",
    "s1_train.py",
    "s2_train.py",
    "s2_train_v3.py",
    "s2_train_v3_lora.py",
    "download.py",
    "onnx_export.py",
    "export_torch_script.py",
    "export_torch_script_v3v4.py",
    "stream_v2pro.py",
    # Note: utils.py must not be skipped — feature_extractor/cnhubert.py imports utils at top level
}
SKIP_TOP_DIRS = {"prepare_datasets", "__pycache__"}


def _ignore_heavy(cur, names):
    """copytree ignore: skip caches and heavy-weight dirs that will be provided by junctions."""
    rel = Path(cur).relative_to(SOURCE_ROOT / "GPT_SoVITS")
    out = set()
    for n in names:
        if n == "__pycache__" or n.endswith(".pyc"):
            out.add(n)
        p = rel / n
        if p.as_posix() in {"text/G2PWModel",
                            "pretrained_models/sv",
                            "pretrained_models/fast_langdetect",
                            "pretrained_models/gsv-v4-pretrained"}:
            out.add(n)
    return out


def _copy_tree(src: Path, dst: Path, ignore=None) -> None:
    if dst.exists():
        return  # already exists, treat as done (idempotent)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, ignore=ignore, dirs_exist_ok=True)


def copy_vendor() -> None:
    """Copy the inference-needed source to vendor/ once (except heavy-weight dirs)."""
    src_pkg = SOURCE_ROOT / "GPT_SoVITS"
    if not src_pkg.is_dir():
        raise FileNotFoundError(
            f"未找到 GPT-SoVITS 源码: {src_pkg}（可用环境变量 GSV_SOURCE_ROOT 指定）"
        )

    # GPT_SoVITS/ package body (with top-level file filtering)
    if not PACKAGE_DIR.is_dir():
        PACKAGE_DIR.mkdir(parents=True)
    for name in sorted(os.listdir(src_pkg)):
        if name in SKIP_TOP_FILES or name in SKIP_TOP_DIRS:
            continue
        src = src_pkg / name
        dst = PACKAGE_DIR / name
        if src.is_dir():
            _copy_tree(src, dst, ignore=_ignore_heavy)
        else:
            if not dst.exists():
                shutil.copy2(src, dst)

    # tools/ (repo root, for `from tools.i18n...` / `from tools.audio_sr...`)
    src_tools = SOURCE_ROOT / "tools"
    if src_tools.is_dir():
        _copy_tree(src_tools, TOOLS_DIR,
                   ignore=lambda d, n: {x for x in n if x == "__pycache__"})


def default_links() -> dict[Path, Path]:
    """Default junction map: vendor-internal links → models/ targets (can be overridden by env).

    Target dirs are derived from the corresponding fields of paths.resolve_config, keeping a single source of truth.
    """
    from . import paths

    cfg = paths.resolve_config({})
    links = {
        PACKAGE_DIR / "text/G2PWModel": cfg["g2pw_dir"],
        PACKAGE_DIR / "pretrained_models/sv": cfg["sv_dir"],
        PACKAGE_DIR / "pretrained_models/fast_langdetect": cfg["langdetect_dir"],
        PACKAGE_DIR / "pretrained_models/gsv-v4-pretrained": cfg["vocoder_dir"],
    }
    return {k: v for k, v in links.items() if v is not None}


def _is_link(path: Path) -> bool:
    """Return whether path is a link (on Windows os.path.islink does not recognize junctions; check the reparse attribute instead)."""
    if sys.platform == "win32":
        try:
            st = os.lstat(path)
            return bool(st.st_file_attributes & 0x400)  # FILE_ATTRIBUTE_REPARSE_POINT
        except OSError:
            return False
    return os.path.islink(str(path))


def _remove_link_dir(path: Path) -> bool:
    """Remove an existing junction/symlink itself (never follow into the target)."""
    if not path.exists() and not _is_link(path):
        return True
    if _is_link(path):
        try:
            os.rmdir(path)  # junctions/symlinks can be removed by rmdir on Windows
            return True
        except OSError:
            return False
    return False  # real dir: leave it (means a previous run fell back to a physical copy)


def _make_junction(link: Path, target: Path) -> bool:
    if sys.platform == "win32":
        r = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            capture_output=True,  # mklink output is in the local encoding (GBK); do not decode as text
        )
        return r.returncode == 0 and link.exists()
    try:
        os.symlink(target, link, target_is_directory=True)
        return True
    except OSError:
        return False


def ensure_links(links: dict[Path, Path] | None = None) -> list[tuple[Path, str]]:
    """Ensure each junction exists and is valid; fall back to a physical copy on failure.

    Returns [(link, "junction"|"copy"|"ok"), ...] for logs/diagnostics.
    """
    if links is None:
        links = default_links()
    results = []
    for link, target in links.items():
        target = Path(target)
        if not target.is_dir():
            results.append((link, f"skip(target missing: {target})"))
            continue
        # Validity check: link exists and is not dangling (probe with the first child)
        probe = next(target.iterdir(), None)
        if link.exists() and not _is_link(link):
            results.append((link, "ok(real dir)"))  # a previous run already fell back to a copy
            continue
        if link.exists():
            valid = probe is None or (link / probe.name).exists()
            if valid:
                results.append((link, "ok(junction)"))
                continue
            _remove_link_dir(link)
        if _make_junction(link, target):
            results.append((link, "junction"))
            continue
        # Fallback: physical copy
        try:
            shutil.copytree(target, link, dirs_exist_ok=False)
            results.append((link, "copy"))
        except OSError as e:
            results.append((link, f"failed({e})"))
    return results


def ensure_vendor(auto_copy: bool = True) -> None:
    """Engine init entry: ensure vendor code exists + weight links are valid (idempotent)."""
    if not (PACKAGE_DIR / "TTS_infer_pack" / "TTS.py").exists():
        if not auto_copy:
            raise FileNotFoundError(
                f"vendor 代码缺失: {PACKAGE_DIR}（运行 python -m core.gsv.vendor_links 复制）"
            )
        copy_vendor()
    ensure_links()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="复制 GPT-SoVITS 推理源码到 vendor 并建立权重链接")
    ap.add_argument("--copy-only", action="store_true", help="仅复制源码，不建链接")
    args = ap.parse_args()

    print(f"source : {SOURCE_ROOT}")
    print(f"vendor : {VENDOR_ROOT}")
    if not args.copy_only:
        ensure_vendor(auto_copy=True)
        for link, st in ensure_links():
            print(f"  link  {link.relative_to(VENDOR_ROOT)} <- {st}")
    else:
        copy_vendor()
        print("  copy  done")
    print("VENDOR_READY")

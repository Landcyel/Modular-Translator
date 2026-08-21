"""vendor 目录复制与重型权重链接（junction）管理。

布局（vendor 根 = 本文件所在目录下的 vendor/）::

    vendor/
    ├── tools/                  # 从 GPT-SoVITS-main/tools 复制（i18n/audio_sr 等）
    └── GPT_SoVITS/             # 从 GPT-SoVITS-main/GPT_SoVITS 复制（仅推理所需）
        ├── AR/ module/ text/ TTS_infer_pack/ BigVGAN/ feature_extractor/
        ├── f5_tts/ eres2net/ configs/
        ├── sv.py process_ckpt.py
        ├── text/G2PWModel/                        → junction → models/gsv/g2pw/G2PWModel
        └── pretrained_models/
            ├── sv/                                → junction → models/gsv/sv
            ├── fast_langdetect/                   → junction → models/gsv/fast_langdetect
            └── gsv-v4-pretrained/                 → junction → models/v4/gsv-v4-pretrained

原则:
- vendored 源码与上游逐字节一致（零修改），import 靠 sys.path + CWD=vendor 根解析
  （上游同样依赖 CWD，见 GPT-SoVITS-main/GPT_SoVITS/TTS_infer_pack/TTS.py 顶部）
- 重型权重不重复占盘: 用 Windows junction（mklink /J，免管理员）指向 models/；
  junction 创建失败自动回退物理拷贝（copytree）
- 本模块幂等，可反复调用
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

# GPT_SoVITS/ 顶层不复制（训练/旧 GUI/导出，推理不需要）
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
    # 注意: utils.py 不能排除 —— feature_extractor/cnhubert.py 顶层 import utils
}
SKIP_TOP_DIRS = {"prepare_datasets", "__pycache__"}


def _ignore_heavy(cur, names):
    """copytree ignore: 跳过缓存与将由 junction 提供的重权重目录。"""
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
        return  # 已存在视为完成（幂等）
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, ignore=ignore, dirs_exist_ok=True)


def copy_vendor() -> None:
    """一次性复制推理所需源码到 vendor/（重权重目录除外）。"""
    src_pkg = SOURCE_ROOT / "GPT_SoVITS"
    if not src_pkg.is_dir():
        raise FileNotFoundError(
            f"未找到 GPT-SoVITS 源码: {src_pkg}（可用环境变量 GSV_SOURCE_ROOT 指定）"
        )

    # GPT_SoVITS/ 包本体（含顶层文件过滤）
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

    # tools/（repo 根，供 `from tools.i18n...` / `from tools.audio_sr...`）
    src_tools = SOURCE_ROOT / "tools"
    if src_tools.is_dir():
        _copy_tree(src_tools, TOOLS_DIR,
                   ignore=lambda d, n: {x for x in n if x == "__pycache__"})


def default_links() -> dict[Path, Path]:
    """默认 junction 映射: vendor 内链接 → models/ 目标（可被 env 覆盖）。

    目标目录由 paths.resolve_config 的对应字段推导，保持单一事实来源。
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
    """判断是否为链接（Windows 上 os.path.islink 不识别 junction，需查 reparse 属性）。"""
    if sys.platform == "win32":
        try:
            st = os.lstat(path)
            return bool(st.st_file_attributes & 0x400)  # FILE_ATTRIBUTE_REPARSE_POINT
        except OSError:
            return False
    return os.path.islink(str(path))


def _remove_link_dir(path: Path) -> bool:
    """删除已存在的 junction/symlink 本身（绝不跟随进目标）。"""
    if not path.exists() and not _is_link(path):
        return True
    if _is_link(path):
        try:
            os.rmdir(path)  # junction/symlink 在 Windows 上可被 rmdir 摘除
            return True
        except OSError:
            return False
    return False  # 真实目录：不动（说明上次回退成了物理拷贝）


def _make_junction(link: Path, target: Path) -> bool:
    if sys.platform == "win32":
        r = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            capture_output=True,  # mklink 输出为本地编码（GBK），不做文本解码
        )
        return r.returncode == 0 and link.exists()
    try:
        os.symlink(target, link, target_is_directory=True)
        return True
    except OSError:
        return False


def ensure_links(links: dict[Path, Path] | None = None) -> list[tuple[Path, str]]:
    """确保每条 junction 存在且有效；失败回退物理拷贝。

    返回 [(link, "junction"|"copy"|"ok"), ...] 供日志/诊断。
    """
    if links is None:
        links = default_links()
    results = []
    for link, target in links.items():
        target = Path(target)
        if not target.is_dir():
            results.append((link, f"skip(target missing: {target})"))
            continue
        # 有效校验: 链接存在且非悬空（取第一个子项探测）
        probe = next(target.iterdir(), None)
        if link.exists() and not _is_link(link):
            results.append((link, "ok(real dir)"))  # 上次已回退拷贝
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
        # 回退: 物理拷贝
        try:
            shutil.copytree(target, link, dirs_exist_ok=False)
            results.append((link, "copy"))
        except OSError as e:
            results.append((link, f"failed({e})"))
    return results


def ensure_vendor(auto_copy: bool = True) -> None:
    """引擎初始化入口: 保证 vendor 代码存在 + 权重链接有效（幂等）。"""
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

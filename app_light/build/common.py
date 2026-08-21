"""Modular Translator 打包共享逻辑（CPU/CUDA 双版本共用，v3）。

本模块由 build/build_cpu.py 与 build/build_cuda.py 共同调用，
实现 PyInstaller onedir 绿色目录打包的完整流程（构建 -> 资源拷贝 -> 校验 -> 冒烟）。

相对旧版 build_package.py 的关键修正（基于打包产物运行日志实证的导入错误）：
1. 解释器锁定：必须用项目 .venv 运行构建，杜绝全局 Python 混入旧版依赖
   （曾把 numpy 2.4.6 混入产物，导致 _multiarray_umath DLL 加载失败）；
2. vendor 导入自动扫描：AST 解析 core/gsv/vendor 全部 import 生成 hidden imports，
   不再依赖手写清单（曾漏收 ffmpeg-python 导致运行时 "No module named 'ffmpeg'"）；
3. --collect-data soundfile：收集 libsndfile_x64.dll（曾缺失导致 import soundfile 失败）；
4. 不再排除 faster_whisper：vendor ASR 工具顶层导入它，排除会导致 ImportError；
5. PyInstaller 中间产物隔离到 build/pyinstaller-<mode>/，不污染项目根。

本脚本零第三方依赖（仅标准库），须用项目 .venv 的 Python 运行。
"""

from __future__ import annotations

import ast
import argparse
import os
import shutil
import stat
import subprocess
import sys
import time
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# 路径常量
# ---------------------------------------------------------------------------

# build/common.py -> 项目根（脚本目录不固定：app_light/build/ 或分支根 build/）
# 从脚本所在目录向上找第一个同时含 configs/ 与 APP.py 的应用根
def _find_app_root(start: Path) -> Path:
    for cand in (start, *start.parents):
        if (cand / "configs").is_dir() and (cand / "APP.py").is_file():
            return cand
    raise RuntimeError(f"无法定位项目根（未找到含 configs/ 与 APP.py 的目录）: {start}")


ROOT = _find_app_root(Path(__file__).resolve().parent)
BUILD_ASSETS = ROOT / "build_assets"
ENGINE_ZIP = BUILD_ASSETS / "flet-windows.zip"
APP_ENTRY = ROOT / "APP.py"
LOGO_FILE = ROOT / "material" / "logo.png"
PROJECT_VENV = ROOT / ".venv"
VENV_PYTHON = PROJECT_VENV / "Scripts" / "python.exe"

# 产物根需外置的用户可编辑/品牌内容
COPY_DIRS = ["configs", "material"]

# 仅复刻目录结构（空目录、不拷贝文件）的运行期目录
STRUCTURE_DIRS = ["dependencies", "output", "temp", "logs", "characters"]

# 依赖源目录
VENDOR_DIR = ROOT / "core" / "gsv" / "vendor"
MOSS_SRC = ROOT / "dependencies" / "MOSS-Transcribe-Diarize"
RUNTIME_SRC = ROOT / "dependencies" / "runtime"
LLAMA_SRC = ROOT / "dependencies" / "llama-release"
FFMPEG_SRC = ROOT / "dependencies" / "FFmpeg"
WHISPER_CUDA_SRC = ROOT / "dependencies" / "fasterwisper-cuda"
DEPS_VENV_SP = ROOT / "dependencies" / "venv" / "Lib" / "site-packages"

# PyInstaller 中间产物隔离目录（spec/work/dist 全部落在 build/ 下）
PYINSTALLER_BUILD_DIR = ROOT / "build" / "pyinstaller"

# ---------------------------------------------------------------------------
# 依赖清单
# ---------------------------------------------------------------------------

# 运行期动态导入/标准库兜底，必须无条件加入 hidden imports（audioop 由
# audioop-lts 提供，Python 3.13+ 已从标准库移除，pydub 依赖）
EXTRA_HIDDEN_IMPORTS = [
    "audioop",                                       # pydub
    "pickletools",                                   # GSV torch.load 链路
    "transformers.configuration_utils",              # PretrainedConfig
    "transformers.models.auto.configuration_auto",   # AutoConfig 动态解析
    "transformers.generation.configuration_utils",   # 生成配置兜底
]

# MOSS 本地包主链路模块（不收集 web_cli/server 以减体积）
MOSS_HIDDEN_IMPORTS = [
    "moss_transcribe_diarize",
    "moss_transcribe_diarize.app.model_runner",
    "moss_transcribe_diarize.subtitle",
]

# 统一排除：torch 系由外挂 runtime 提供（app/torch_runtime.py 运行时注入）；
# 其余确认无运行时引用（含 vendor 顶层导入）。
# 警示：列表内任何包若被 core/gsv/vendor 顶层 import，即为运行时炸弹
# （曾误排除 matplotlib，lr_schedulers.py 顶层导入导致 GSV 服务启动失败）。
EXCLUDES = [
    "torch",
    "torchaudio",
    "pytest",
    "flet_web",
    "hf_xet",
]

# 多依赖源时版本必须一致的关键包（混装会导致二进制 DLL 加载失败）
KEY_PACKAGES = ["numpy", "scipy", "av", "soundfile", "transformers", "librosa"]

# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------


def step(msg: str) -> None:
    print(f"\n== {msg} ==")


def _rmtree_readonly(func, path, exc) -> None:
    """rmtree 遇到只读文件时去掉只读属性并重试。"""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def force_rmtree(path: Path) -> None:
    """删除目录树，自动处理 Windows 只读文件。"""
    shutil.rmtree(path, onexc=_rmtree_readonly)


# 重建时保留的产物顶层目录（用户资产：角色/模型，体积大且重新部署成本高）
KEEP_OUTPUT_DIRS = {"characters", "dependencies"}
# dependencies/ 下由打包脚本管理的子目录（清理时删除，按当前模式重拷）
MANAGED_DEPS_SUBDIRS = {"FFmpeg", "llama-release"}
# runtime/ 下由打包脚本管理的子目录（同上）
MANAGED_RUNTIME_SUBDIRS = {"torch-cpu", "torch-cuda"}


def _clean_managed_deps(deps_dir: Path) -> None:
    """删除 dependencies/ 下受管子目录；models/、MOSS-Transcribe-Diarize/、
    venv/、requirements/、fasterwisper-cuda/（用户可选放置 DLL）等保留。"""
    for name in MANAGED_DEPS_SUBDIRS:
        p = deps_dir / name
        if p.exists():
            force_rmtree(p)
    runtime = deps_dir / "runtime"
    if runtime.is_dir():
        for name in MANAGED_RUNTIME_SUBDIRS:
            p = runtime / name
            if p.exists():
                force_rmtree(p)


def clean_output_dir(dst_dir: Path) -> None:
    """清理旧产物，但保留用户已放置的 characters/ 与 dependencies/ 资产。

    dependencies/ 下受管子目录（FFmpeg / llama-release / runtime/torch-*）
    一并删除，由本次构建按当前模式重拷；模型、角色、可选 DLL 等非受管
    内容原样保留，避免重建后需重新部署 GB 级资产。
    """
    if not dst_dir.exists():
        return
    for child in dst_dir.iterdir():
        if child.name in KEEP_OUTPUT_DIRS:
            if child.name == "dependencies":
                _clean_managed_deps(child)
            continue
        if child.is_dir():
            force_rmtree(child)
        else:
            os.chmod(child, stat.S_IWRITE)
            child.unlink()


# PyInstaller hook 目录（官方 + contrib），用于判断库是否已有收集 hook
PYINSTALLER_HOOK_DIR = PROJECT_VENV / "Lib" / "site-packages" / "PyInstaller" / "hooks"
CONTRIB_HOOK_DIR = (
    PROJECT_VENV / "Lib" / "site-packages" / "_pyinstaller_hooks_contrib" / "stdhooks"
)


def has_pyinstaller_hook(name: str, site_packages: list[Path]) -> bool:
    """判断库是否有 PyInstaller hook（官方 / contrib / 包内 __pyinstaller 目录）。

    无 hook 的库（如 jieba_fast）子模块/数据文件默认收集不完整，
    运行时 No module named '<pkg>.sub' 或数据缺失，需 --collect-all 兜底。
    """
    if (PYINSTALLER_HOOK_DIR / f"hook-{name}.py").is_file():
        return True
    if (CONTRIB_HOOK_DIR / f"hook-{name}.py").is_file():
        return True
    # 包内自带 hook（如 pypinyin/__pyinstaller/hook-pypinyin.py）
    return any((sp / name / "__pyinstaller").is_dir() for sp in site_packages)


def require_project_venv() -> None:
    """强制使用项目 .venv 的解释器运行构建。

    旧版脚本允许任意解释器（甚至 PATH 上的全局 Python），曾导致 numpy 2.4.6
    混入产物、_multiarray_umath DLL 加载失败。此校验为硬约束。
    """
    exe = Path(sys.executable).resolve()
    if not exe.is_relative_to(PROJECT_VENV.resolve()):
        sys.exit(
            "[错误] 必须使用项目 .venv 的 Python 运行构建脚本。\n"
            f"  当前解释器: {exe}\n"
            f"  正确用法: {VENV_PYTHON} build/build_cpu.py\n"
            "  使用全局 Python 会把旧版依赖混入产物，导致运行时 DLL 加载失败。"
        )
    if not (PROJECT_VENV / "Lib" / "site-packages" / "PyInstaller").is_dir():
        sys.exit(
            "[错误] 项目 .venv 未安装 PyInstaller。\n"
            f"  请先执行: {VENV_PYTHON} -m pip install pyinstaller"
        )


def find_site_packages(extra_paths: bool) -> list[Path]:
    """返回依赖源 site-packages 列表。

    主源固定为项目 .venv（保证 numpy 等二进制包版本正确）；dependencies/venv
    仅在显式 --extra-paths 时追加（多源时由版本一致性预检把关）。
    """
    result: list[Path] = []
    main_sp = PROJECT_VENV / "Lib" / "site-packages"
    if main_sp.is_dir():
        result.append(main_sp)
    if extra_paths and DEPS_VENV_SP.is_dir() and DEPS_VENV_SP != main_sp:
        result.append(DEPS_VENV_SP)
    return result


def package_version(site_packages: Path, name: str) -> str | None:
    """读取指定 site-packages 中某包的 dist-info 版本。"""
    for dist in site_packages.glob(f"{name}-*.dist-info"):
        meta = dist / "METADATA"
        if meta.is_file():
            for line in meta.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.startswith("Version:"):
                    return line.split(":", 1)[1].strip()
    return None


def check_key_package_consistency(site_packages: list[Path]) -> None:
    """多依赖源时关键包版本必须一致，防止混装导致 DLL 加载失败。"""
    if len(site_packages) < 2:
        return
    for name in KEY_PACKAGES:
        versions: dict[str, Path] = {}
        for sp in site_packages:
            v = package_version(sp, name)
            if v is not None:
                versions.setdefault(v, sp)
        if len(versions) > 1:
            detail = ", ".join(f"{v}({sp})" for v, sp in versions.items())
            sys.exit(
                f"[错误] 关键包 {name} 在多个依赖源中版本不一致: {detail}\n"
                "  混装会导致运行时二进制 DLL 加载失败，请统一版本后重试。"
            )


def import_available(name: str, site_packages: list[Path]) -> bool:
    """判断顶层模块在任一依赖源中是否存在（用于可选依赖探测）。"""
    for sp in site_packages:
        if (sp / name).is_dir():
            return True
        if (sp / f"{name}.py").is_file():
            return True
        if (sp / f"{name}.pyd").exists():
            return True
    return False


# ---------------------------------------------------------------------------
# TorchScript 源码化（jit.script 编译需要 Python 源码）
# ---------------------------------------------------------------------------

# 不源码化的库：外挂 torch 系 / 二进制大库（jit 编译图不会引用其 Python 函数源码，
# PyInstaller 正常收集进 PYZ 即可）。einops 除外——x_transformers 内部调用其
# Python 函数（rearrange 等），可能进入 jit 编译图，须源码化。
SOURCE_SKIP = {
    "torch", "torchaudio", "numpy", "scipy", "av", "soundfile", "onnxruntime",
    "ctranslate2", "matplotlib", "PIL", "sklearn", "pandas", "sympy", "numba",
    "llvmlite", "torchmetrics", "pytorch_lightning", "transformers", "peft",
    "librosa", "openai", "fastapi", "uvicorn", "pydantic", "huggingface_hub",
    "safetensors", "tqdm", "requests", "packaging", "loguru", "regex", "jieba",
    "jieba_fast", "nltk", "g2p_en", "pyopenjtalk", "cn2an", "pypinyin", "opencc",
    "inflect", "wordsegment", "typeguard", "yaml", "fast_langdetect", "split_lang",
}

# 源码化种子：GSV 的 jit.script 编译图引用 x_transformers 的 Python 函数
# （softclamp 等），其依赖链（einx/frozendict/einops/torch_einops_utils）由
# expand_source_closure 递归展开，一次到位。
SOURCE_SEEDS = ["x_transformers"]


def _scan_pkg_imports(pkg_dir: Path) -> set[str]:
    """AST 扫描某第三方包内全部 import，返回顶层包名集合。"""
    deps: set[str] = set()
    for py in pkg_dir.rglob("*.py"):
        try:
            tree = ast.parse(py.read_text(encoding="utf-8", errors="ignore"))
        except (SyntaxError, OSError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    deps.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                deps.add(node.module.split(".")[0])
    return deps


def expand_source_closure(site_packages: list[Path], seeds: list[str]) -> list[str]:
    """从种子库递归展开第三方 import 闭包，返回需源码化的包名（有序去重）。

    ``--exclude-module`` 后 PyInstaller 不再分析被排除库的 import，其依赖链
    必须在此显式覆盖——否则运行时 "No module named" 逐层暴露（x_transformers
    → einx → frozendict）。递归扫描保证一次到位。
    """
    stdlib = set(sys.stdlib_module_names)
    closure: set[str] = set()
    queue = list(seeds)
    while queue:
        name = queue.pop()
        if name in closure or name in stdlib or name in SOURCE_SKIP:
            continue
        src = next((sp / name for sp in site_packages if (sp / name).is_dir()), None)
        if src is None:
            # 单文件模块同样源码化
            if any((sp / f"{name}.py").is_file() for sp in site_packages):
                closure.add(name)
            continue
        closure.add(name)
        for dep in sorted(_scan_pkg_imports(src)):
            if dep not in stdlib and dep not in SOURCE_SKIP:
                queue.append(dep)
    return sorted(closure)


def scan_vendor_imports(site_packages: list[Path]) -> list[str]:
    """AST 扫描 core/gsv/vendor 下全部 import，自动生成 hidden imports。

    vendor 代码以 --add-data 打入产物、运行时经 sys.path 动态加载，
    PyInstaller 静态分析不可见，必须显式 hidden-import。手写清单容易漏收
    （曾漏 ffmpeg-python），改为扫描后与手工白名单合并。
    返回仅含已在依赖源中存在的顶层包名。
    """
    if not VENDOR_DIR.is_dir():
        print(f"  [警告] 未找到 GSV vendor 目录: {VENDOR_DIR}（打包后语音合成不可用）")
        return []

    internal_tops = {d.name for d in VENDOR_DIR.iterdir() if d.is_dir()}
    stdlib = set(sys.stdlib_module_names)
    tops: set[str] = set()

    for py in VENDOR_DIR.rglob("*.py"):
        try:
            tree = ast.parse(py.read_text(encoding="utf-8", errors="ignore"))
        except (SyntaxError, OSError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    tops.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                tops.add(node.module.split(".")[0])

    result = []
    skipped: list[str] = []
    for top in sorted(tops):
        if top in stdlib or top in EXCLUDES or top in internal_tops:
            continue
        if import_available(top, site_packages):
            result.append(top)
        else:
            skipped.append(top)
    if skipped:
        print(f"  [跳过] vendor 引用的可选依赖未安装（不影响主链路）: {', '.join(skipped)}")
    return result


# ---------------------------------------------------------------------------
# 引擎包（flet 桌面引擎）准备
# ---------------------------------------------------------------------------


def find_local_engine_cache() -> Path | None:
    """在本机 ~/.flet/client/ 下定位引擎缓存目录。"""
    client_dir = Path.home() / ".flet" / "client"
    if not client_dir.is_dir():
        return None
    for d in sorted(client_dir.iterdir(), reverse=True):
        if d.is_dir() and d.name.startswith("flet-desktop-"):
            if (d / "flet" / "flet.exe").is_file():
                return d
    return None


def check_engine_zip(zip_path: Path) -> None:
    """校验引擎包：结构 + 完整性。"""
    with zipfile.ZipFile(zip_path) as zf:
        if "flet/flet.exe" not in zf.namelist():
            sys.exit(f"[错误] 引擎包 {zip_path} 缺少 flet/flet.exe，结构不符")
        bad = zf.testzip()
        if bad is not None:
            sys.exit(f"[错误] 引擎包损坏: {bad}")


def build_engine_zip_from_cache(cache_dir: Path) -> Path:
    """把本机引擎缓存目录重新打包为 build_assets/flet-windows.zip。"""
    BUILD_ASSETS.mkdir(parents=True, exist_ok=True)
    shutil.make_archive(str(ENGINE_ZIP.with_suffix("")), "zip", root_dir=str(cache_dir))
    if not ENGINE_ZIP.exists():
        sys.exit(f"[错误] 重新打包引擎失败: {ENGINE_ZIP} 未生成")
    return ENGINE_ZIP


def prepare_engine(embed_engine: bool) -> Path | None:
    """准备引擎包。返回内嵌 zip 路径；--no-engine 时返回 None。"""
    if not embed_engine:
        step("1/8 引擎外置模式（--no-engine）")
        print("  跳过内嵌引擎；首次运行将从 GitHub 下载 flet-windows.zip 到 ~/.flet/client/")
        return None

    step("1/8 准备引擎包")
    if ENGINE_ZIP.exists():
        print(f"  引擎包已存在，校验: {ENGINE_ZIP}")
        check_engine_zip(ENGINE_ZIP)
    else:
        cache = find_local_engine_cache()
        if cache is None:
            sys.exit(
                f"[错误] 缺少引擎包 {ENGINE_ZIP}，且本机 ~/.flet/client/ 下无引擎缓存。\n"
                "  处理方式：\n"
                "    1) 手动放置官方 flet-windows.zip（flet 0.86.2）到 build_assets/；或\n"
                "    2) 先在本机运行一次应用生成缓存（~/.flet/client/flet-desktop-full-0.86.2/）；或\n"
                "    3) 使用 --no-engine 跳过内嵌（引擎外置，首次运行联网下载，产物小 ~40MB）。"
            )
        print(f"  未找到引擎包，自动从本机缓存重新打包: {cache}")
        engine = build_engine_zip_from_cache(cache)
        check_engine_zip(engine)
        print(f"  已生成 {engine.name} ({engine.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"  引擎包 OK: {ENGINE_ZIP.name} ({ENGINE_ZIP.stat().st_size / 1024 / 1024:.1f} MB)")
    return ENGINE_ZIP


# ---------------------------------------------------------------------------
# PyInstaller 构建
# ---------------------------------------------------------------------------


def build_pyinstaller_cmd(
    mode: str,
    engine_zip: Path | None,
    site_packages: list[Path],
    vendor_imports: list[str],
    source_closure: list[str],
    extra_paths: bool,
) -> list[str]:
    """组装 PyInstaller 命令（CPU/CUDA 共用，依赖差异在资源拷贝阶段）。

    --paths 不显式加当前解释器的 site-packages（PyInstaller 自动搜索运行环境，
    显式添加会触发 "Foreign Python environment" deprecation 警告）；仅加
    MOSS 源码等非标准路径，以及 --extra-paths 时追加的 dependencies/venv。
    site_packages 列表仅用于可选依赖探测与版本一致性预检。
    """
    build_dir = PYINSTALLER_BUILD_DIR / mode
    cmd = [
        str(sys.executable), "-m", "PyInstaller",
        str(APP_ENTRY),
        "--name", "ModularTranslator",
        "--distpath", str(build_dir / "dist"),
        "--workpath", str(build_dir / "work"),
        "--specpath", str(build_dir),
        "--onedir",
        "--noconsole",
        "--noconfirm",
        "--clean",
        "--collect-all", "flet",
        "--collect-all", "flet_desktop",   # flet 库内延迟导入，静态分析收不到
        "--collect-all", "pydub",
        "--collect-data", "budoux",        # split_lang 日文分词依赖；纯 Python 包内
                                           # 数据文件（skip_nodes.json/models/*.json）
                                           # PyInstaller 默认不收集，须显式收集
    ]

    if extra_paths:
        cmd += ["--paths", str(DEPS_VENV_SP)]
        print(f"  [路径] {DEPS_VENV_SP}（--extra-paths）")

    if MOSS_SRC.is_dir():
        cmd += ["--paths", str(MOSS_SRC)]
        cmd += ["--collect-data", "moss_transcribe_diarize"]
        for mod in MOSS_HIDDEN_IMPORTS:
            cmd += ["--hidden-import", mod]
        print(f"  [MOSS] {MOSS_SRC}")
    else:
        print(f"  [警告] 未找到 MOSS 源码目录: {MOSS_SRC}（打包后转写可能不可用）")

    if VENDOR_DIR.is_dir():
        cmd += ["--add-data", f"{VENDOR_DIR};core/gsv/vendor"]
        print(f"  [GSV vendor] {VENDOR_DIR} -> _internal/core/gsv/vendor")

    # TorchScript（torch.jit.script）编译需要 Python 源码（inspect.getsource），
    # PyInstaller 6.x 无 --collect-source，PYZ 字节码无源码。source_closure 内的
    # 库统一：exclude 模块收集 + add-data 源码目录打入 _internal，运行时经
    # sys._MEIPASS（frozen 下 _internal 在 sys.path）从源码加载。
    # 注意：exclude 后 PyInstaller 不再分析其 import，依赖链由
    # expand_source_closure 递归覆盖（x_transformers → einx → frozendict 等）。
    for name in source_closure:
        src = next(
            (sp / name for sp in site_packages if (sp / name).is_dir()),
            None,
        )
        if src is None:
            src = next(
                (sp / f"{name}.py" for sp in site_packages if (sp / f"{name}.py").is_file()),
                None,
            )
            if src is None:
                print(f"  [警告] 未找到 {name} 源码，TorchScript 编译可能失败")
                continue
            cmd += ["--exclude-module", name]
            cmd += ["--add-data", f"{src};{name}.py"]
            print(f"  [源码化] {name}（单文件）-> _internal/{name}.py")
        else:
            cmd += ["--exclude-module", name]
            cmd += ["--add-data", f"{src};{name}"]
            print(f"  [源码化] {name} -> _internal/{name}/")

    # 无 PyInstaller hook 的库统一 collect-all：子模块/数据/二进制一次收集，
    # 防止运行时 No module named '<pkg>.sub'（jieba_fast.posseg 教训）或数据缺失。
    # 源码化闭包内的库除外（已 exclude + add-data 全量拷贝）。
    for mod in vendor_imports:
        if mod in source_closure:
            continue
        if not has_pyinstaller_hook(mod, site_packages):
            cmd += ["--collect-all", mod]
            print(f"  [collect-all] {mod}（无官方 hook，兜底全量收集）")

    for mod in EXTRA_HIDDEN_IMPORTS:
        cmd += ["--hidden-import", mod]
        print(f"  [隐藏导入] {mod}")

    for mod in vendor_imports:
        cmd += ["--hidden-import", mod]
        print(f"  [vendor 扫描] {mod}")

    if LOGO_FILE.exists():
        cmd += ["--icon", str(LOGO_FILE)]
        print(f"  [Logo] 使用 {LOGO_FILE} 作为 exe 图标")
    else:
        print(f"  [警告] 未找到 Logo: {LOGO_FILE}，跳过 --icon")

    if engine_zip is not None:
        cmd += ["--add-data", f"{engine_zip};flet_desktop/app"]

    for mod in EXCLUDES:
        cmd += ["--exclude-module", mod]
    return cmd


# ---------------------------------------------------------------------------
# 资源拷贝
# ---------------------------------------------------------------------------


def create_structure_dirs(src_root: Path, dst_root: Path, top_dirs: list[str]) -> None:
    """在 dst_root 下复刻 src_root 中 top_dirs 的目录结构（仅建空目录）。"""
    for name in top_dirs:
        src = src_root / name
        dst = dst_root / name
        if not src.is_dir():
            print(f"  [警告] 源目录缺失，跳过结构复刻: {src}")
            continue
        created = 0
        for dirpath, _dirnames, _filenames in os.walk(src):
            rel = Path(dirpath).relative_to(src)
            target = dst if str(rel) == "." else dst / rel
            target.mkdir(parents=True, exist_ok=True)
            created += 1
        print(f"  {name}/ 结构 -> {dst}（{created} 个目录）")


def generate_icon(dst_dir: Path) -> None:
    """构建期显式生成产物 material/logo.ico（PIL 从 logo.png 转换）。

    flet 0.86.2 Window.icon 仅支持 .ico；构建期生成使产物自带图标文件，
    运行时不依赖 PIL/临时目录。
    """
    png = dst_dir / "material" / "logo.png"
    ico = dst_dir / "material" / "logo.ico"
    if not png.is_file():
        print(f"  [警告] 未找到 logo.png，跳过图标生成: {png}")
        return
    try:
        from PIL import Image
        Image.open(str(png)).save(
            str(ico), format="ICO",
            sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
        )
        print(f"  [图标] 显式生成 {ico}（{ico.stat().st_size / 1024:.0f} KB）")
    except Exception as exc:
        print(f"  [警告] 图标生成失败（跳过，运行时可回退）: {exc}")


def copy_configs(dst_dir: Path) -> None:
    step("4/8 拷贝 configs/ + material/ 到产物根")
    for name in COPY_DIRS:
        src = ROOT / name
        dst = dst_dir / name
        if not src.is_dir():
            print(f"  [警告] 源目录缺失，跳过: {src}")
            continue
        if dst.exists():
            force_rmtree(dst)
        shutil.copytree(src, dst)
        print(f"  {name}/ -> {dst}")
    generate_icon(dst_dir)  # material 拷贝后显式生成任务栏图标


def copy_ffmpeg(dst_dir: Path) -> None:
    """只拷贝 ffmpeg.exe + ffprobe.exe，跳过 ffplay.exe（Windows 不需要）。"""
    step("5/8 拷贝精简 FFmpeg（仅 ffmpeg + ffprobe）")
    src_dir = FFMPEG_SRC
    dst = dst_dir / "dependencies" / "FFmpeg"
    if not src_dir.is_dir():
        sys.exit(f"[错误] 缺少项目刚需的 FFmpeg 目录: {src_dir}")

    dst.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in ("ffmpeg.exe", "ffprobe.exe"):
        src = src_dir / name
        if not src.exists():
            print(f"  [警告] 缺少 {src}，跳过")
            continue
        shutil.copy2(src, dst / name)
        copied.append(name)
    if "ffmpeg.exe" not in copied:
        sys.exit(f"[错误] 缺少 FFmpeg 主程序: {src_dir / 'ffmpeg.exe'}")
    size_mb = sum(p.stat().st_size for p in dst.glob("*") if p.is_file()) / 1024 / 1024
    print(f"  {src_dir} -> {dst}（{size_mb:.1f} MB，已跳过 ffplay.exe）")


def _is_llama_cpu_file(name: str) -> bool:
    """判断 llama-release 文件是否属于 CPU 运行所需（排除 CUDA 专用文件）。"""
    lower = name.lower()
    if lower.startswith("cublas") or lower.startswith("cudart"):
        return False
    if lower == "ggml-cuda.dll":
        return False
    return True


def copy_llama_release(mode: str, dst_dir: Path) -> None:
    """按模式拷贝 llama-release：cpu 只拷 CPU 子集（约 44MB），cuda 拷完整（约 667MB）。"""
    if mode == "cpu":
        step("5.5/8 拷贝 llama-release CPU 子集")
    else:
        step("5.5/8 拷贝 llama-release 完整版（CPU+CUDA）")

    src_dir = LLAMA_SRC
    dst = dst_dir / "dependencies" / "llama-release"
    if not src_dir.is_dir():
        print(f"  [警告] 未找到 llama-release: {src_dir}（本地翻译将不可用）")
        return

    if dst.exists():
        force_rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    total = 0
    for item in src_dir.iterdir():
        if not item.is_file():
            continue
        if mode == "cpu" and not _is_llama_cpu_file(item.name):
            continue
        shutil.copy2(item, dst / item.name)
        total += item.stat().st_size
    print(f"  {src_dir} -> {dst}（{total / 1024 / 1024:.1f} MB）")


def copy_torch_runtime(mode: str, include_cpu_fallback: bool, dst_dir: Path) -> None:
    """按模式拷贝 torch 运行时（app/torch_runtime.py 运行时自动选择）。"""
    runtime_dst = dst_dir / "dependencies" / "runtime"

    if mode == "cpu":
        step("5.6/8 拷贝 torch-cpu 运行时")
        _copy_runtime_dir("torch-cpu", runtime_dst)
        return

    # cuda 模式
    step("5.6/8 拷贝 torch-cuda 运行时")
    _copy_runtime_dir("torch-cuda", runtime_dst)
    if include_cpu_fallback:
        step("5.6.1/8 拷贝 torch-cpu 作为无 GPU 回退（--with-cpu-fallback）")
        _copy_runtime_dir("torch-cpu", runtime_dst)
    else:
        print("  [提示] 未指定 --with-cpu-fallback，无 GPU 时将由 torch_runtime 降级使用 torch-cuda 以 CPU 模式运行")


def _copy_runtime_dir(name: str, runtime_dst: Path) -> None:
    src = RUNTIME_SRC / name
    dst = runtime_dst / name
    if not src.is_dir():
        print(f"  [警告] 未找到 {src}，跳过；MOSS/GSV 将无法运行")
        return
    if dst.exists():
        force_rmtree(dst)
    shutil.copytree(src, dst)
    size_mb = sum(p.stat().st_size for p in dst.rglob("*") if p.is_file()) / 1024 / 1024
    print(f"  {src} -> {dst}（{size_mb:.1f} MB）")


def copy_whisper_cuda(enable: bool, dst_dir: Path) -> None:
    """CUDA 版可选打入 fasterwisper-cuda/（CTranslate2 的 CUDA DLL，约 1.3GB）。

    未启用时也确保产物中存在该空目录，作为用户后续放置 DLL 的固定位置。
    """
    dst = dst_dir / "dependencies" / "fasterwisper-cuda"
    if not enable:
        dst.mkdir(parents=True, exist_ok=True)
        print("  [提示] 未指定 --with-whisper-cuda，fasterwisper-cuda/ 保持外置（whisper 走 CPU）")
        print(f"  {dst} 空目录已就位，如需 GPU 加速可自行放置 DLL 或加 --with-whisper-cuda 重建")
        return
    step("5.7/8 拷贝 fasterwisper-cuda/（--with-whisper-cuda）")
    src = WHISPER_CUDA_SRC
    if not src.is_dir():
        print(f"  [警告] 源目录缺失，跳过: {src}")
        return
    if dst.exists():
        force_rmtree(dst)
    shutil.copytree(src, dst)
    size_mb = sum(p.stat().st_size for p in dst.rglob("*") if p.is_file()) / 1024 / 1024
    print(f"  {src} -> {dst}（{size_mb:.1f} MB）")


def copy_characters_if_requested(with_characters: bool, dst_dir: Path) -> None:
    if not with_characters:
        print("  [提示] 未指定 --with-characters，characters/ 仅保留空结构")
        return
    step("5.8/8 拷贝 characters/ 角色资产（--with-characters）")
    src = ROOT / "characters"
    dst = dst_dir / "characters"
    if not src.is_dir():
        print(f"  [警告] 源目录缺失，跳过: {src}")
        return
    if dst.exists():
        force_rmtree(dst)
    shutil.copytree(src, dst)
    size_mb = sum(p.stat().st_size for p in dst.rglob("*") if p.is_file()) / 1024 / 1024
    print(f"  {src} -> {dst}（{size_mb:.1f} MB）")


def write_launchers(dst_dir: Path) -> None:
    step("6/8 生成 start.bat + README + 启动.vbs")
    start_bat = dst_dir / "start.bat"
    start_bat.write_text(
        "@echo off\r\n"
        "rem 固定工作目录为产物根，再启动应用（双保险：代码已不依赖 CWD）\r\n"
        "cd /d %~dp0\r\n"
        "ModularTranslator.exe\r\n",
        encoding="utf-8",
    )
    print(f"  {start_bat}")

    readme_src = BUILD_ASSETS / "README.md"
    if readme_src.exists():
        shutil.copyfile(readme_src, dst_dir / "README.md")
        print(f"  {dst_dir / 'README.md'}")
    else:
        print(f"  [警告] 未找到 {readme_src}，跳过 README.md 拷贝")

    vbs_src = BUILD_ASSETS / "启动.vbs"
    if vbs_src.exists():
        shutil.copyfile(vbs_src, dst_dir / "启动.vbs")
        print(f"  {dst_dir / '启动.vbs'}")
    else:
        print(f"  [警告] 未找到 {vbs_src}，跳过 启动.vbs 拷贝")


def size_report(engine_zip: Path | None, mode: str, dst_dir: Path) -> None:
    step("7/8 体积报告")
    total = 0.0
    for p in dst_dir.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    print(f"  模式: {mode}")
    print(f"  {dst_dir}/ 总大小: {total / 1024 / 1024:.1f} MB")
    if engine_zip is not None:
        eng = dst_dir / "_internal" / "flet_desktop" / "app" / ENGINE_ZIP.name
        if eng.exists():
            print(f"  内嵌引擎: {eng} ({eng.stat().st_size / 1024 / 1024:.1f} MB)")
        else:
            print(f"  [警告] 未在 {dst_dir / '_internal' / 'flet_desktop' / 'app'} 找到内嵌引擎 zip")
    else:
        print("  引擎外置模式：未内嵌引擎，首次运行联网下载（见 README 说明）")
    print(f"\n构建完成: {dst_dir / 'ModularTranslator.exe'}")


# ---------------------------------------------------------------------------
# 构建后校验与冒烟测试
# ---------------------------------------------------------------------------


def verify_package(
    mode: str,
    include_whisper_cuda: bool,
    dst_dir: Path,
    source_closure: list[str],
) -> bool:
    """构建后校验关键模块/二进制依赖是否已打入产物（文件级检查）。"""
    step("8/8 校验产物关键依赖")
    ok = True

    # 1. Analysis TOC 关键模块检查
    analysis_toc = PYINSTALLER_BUILD_DIR / mode / "work" / "ModularTranslator" / "Analysis-00.toc"
    toc_text = ""
    if analysis_toc.exists():
        toc_text = analysis_toc.read_text(encoding="utf-8", errors="ignore")
    else:
        print(f"  [错误] 未找到 Analysis TOC: {analysis_toc}")
        ok = False

    must_have_modules = [
        "pickletools",
        "transformers.configuration_utils",
        "transformers.models.auto.configuration_auto",
        "transformers.generation.configuration_utils",
        "moss_transcribe_diarize",
        "numpy",
        "scipy",
        "ffmpeg",
        "soundfile",
        "faster_whisper",
        "matplotlib",  # vendor lr_schedulers.py 顶层导入，曾漏排除导致 GSV 失败
    ]
    for mod in must_have_modules:
        found = f"'{mod}'" in toc_text or f'"{mod}"' in toc_text
        print(f"  [{'OK' if found else '缺失'}] 模块 {mod}")
        if not found:
            ok = False

    # 2. 产物文件级检查（直接验证二进制/数据文件是否在包内）。
    #    纯 Python 模块（ffmpeg/faster_whisper/soundfile 本体）编译进 PYZ，
    #    不落目录，由上方 toc 检查覆盖；此处只查二进制与数据。
    internal = dst_dir / "_internal"
    file_checks: list[tuple[Path, str, bool]] = [
        (internal / "numpy" / "_core", "_multiarray_umath*.pyd", True),      # numpy C 扩展
        (internal / "numpy.libs", "*.dll", True),                            # numpy 依赖 DLL（BLAS）
        (internal / "_soundfile_data", "libsndfile*.dll", True),             # soundfile 数据 DLL（顶层包）
        (internal / "av.libs", "*.dll", True),                               # av 依赖 DLL
        (internal / "moss_transcribe_diarize", None, False),                 # MOSS 包
        (internal / "core" / "gsv" / "vendor", None, False),                 # GSV vendor 数据
        (internal / "budoux", "skip_nodes.json", True),                      # split_lang 分词数据
        (internal / "jieba_fast" / "posseg", "__init__.py", True),           # 文本前端词性标注
        (internal / "jieba_fast", "dict.txt", True),                         # jieba_fast 词典
    ]
    # 源码化闭包内每个库须有源码文件（jit.script 编译依赖）
    for name in source_closure:
        pkg_dir = internal / name
        if (pkg_dir / "__init__.py").exists() or (pkg_dir / f"{name}.py").exists():
            print(f"  [OK] 源码化 {name}")
        else:
            print(f"  [缺失] 源码化 {name}（{pkg_dir}）")
            ok = False
    if mode == "cpu":
        file_checks.append((dst_dir / "dependencies" / "runtime" / "torch-cpu", "torch/__init__.py", True))
    else:
        file_checks.append((dst_dir / "dependencies" / "runtime" / "torch-cuda", "torch/lib/torch_cuda.dll", True))
        if include_whisper_cuda:
            file_checks.append((dst_dir / "dependencies" / "fasterwisper-cuda", "cublas*.dll", True))

    for base, pattern, need_content in file_checks:
        if not base.exists():
            print(f"  [缺失] 目录 {base}")
            ok = False
            continue
        if pattern is None:
            print(f"  [OK] 目录 {base}")
            continue
        hits = list(base.rglob(pattern)) if base.is_dir() else []
        if hits:
            print(f"  [OK] {base} 含 {hits[0].name}")
        elif need_content:
            print(f"  [缺失] {base} 下未找到 {pattern}")
            ok = False
        else:
            print(f"  [OK] 目录 {base}")

    if ok:
        print("  校验通过：关键模块与二进制依赖均已在包内。")
    else:
        print("  校验失败：请根据上方缺失项补充 hidden imports 或数据收集。")
    return ok


def smoke_test(dst_dir: Path, wait_seconds: float = 15.0) -> None:
    """启动产物验证导入无错误：进程存活 + 无新增导入类错误日志。

    应用启动后会初始化服务（MOSS/GSV 等），若打包存在导入问题，
    服务启动失败会写入 logs/app-error-*.log，可直接暴露历史导入错误。
    仅导入类错误（No module named / DLL load failed 等）判定失败；
    模型/数据未部署产生的错误只打印警告（外置模型不随包分发）。
    """
    step("9/8 冒烟测试：启动产物验证导入无错误")
    logs_dir = dst_dir / "logs"
    logs_before = {p.name for p in logs_dir.glob("app-error-*.log")} if logs_dir.is_dir() else set()

    exe = dst_dir / "ModularTranslator.exe"
    if not exe.exists():
        sys.exit(f"[错误] 未找到产物 {exe}")
    proc = subprocess.Popen([str(exe)], cwd=str(dst_dir))
    print(f"  已启动 {exe.name}，等待 {wait_seconds:.0f} 秒观察……")
    time.sleep(wait_seconds)

    if proc.poll() is not None:
        sys.exit(f"[失败] 产物进程提前退出，退出码 {proc.returncode}，请查看 {logs_dir}")

    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()

    import_error_markers = ("No module named", "ImportError", "ModuleNotFoundError", "DLL load failed")
    fatal: list[tuple[str, str]] = []
    for name in sorted({p.name for p in logs_dir.glob("app-error-*.log")} - logs_before):
        body = (logs_dir / name).read_text(encoding="utf-8", errors="ignore").strip()
        if any(marker in body for marker in import_error_markers):
            fatal.append((name, body))
        else:
            print(f"  [警告] 启动期产生非导入类错误日志 {name}（多为模型/数据未部署，非打包问题）")

    if fatal:
        for name, body in fatal:
            print(f"  [错误] 启动期出现导入类错误 {name}:")
            print("    " + body.replace("\n", "\n    "))
        sys.exit("[失败] 启动期存在导入错误，请根据上方内容修复后重新打包。")
    print(f"  [OK] 进程存活 {wait_seconds:.0f} 秒且无导入类错误日志")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--no-engine",
        action="store_true",
        help="引擎外置：不内嵌 flet-windows.zip（产物小 ~40MB，首次运行需联网下载）",
    )
    ap.add_argument(
        "--with-characters",
        action="store_true",
        help="额外把 characters/ 角色资产拷入产物根（默认仅建空目录，体积 GB 级）",
    )
    ap.add_argument(
        "--output-dir",
        default=None,
        help="产物输出目录（默认按版本: D:\\ModularTranslator_cpu / D:\\ModularTranslator_cuda）",
    )
    ap.add_argument(
        "--extra-paths",
        action="store_true",
        help="追加 dependencies/venv 的 site-packages 为依赖源（需与主源关键包版本一致）",
    )
    ap.add_argument(
        "--verify",
        action="store_true",
        help="构建完成后自动校验关键模块/二进制依赖是否已打入",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="构建完成后启动产物约 15 秒，验证进程存活且无新增错误日志",
    )


def build(
    mode: str,
    default_output: str,
    embed_engine: bool,
    with_characters: bool,
    include_cpu_fallback: bool,
    with_whisper_cuda: bool,
    extra_paths: bool,
    verify: bool,
    smoke: bool,
    output_dir: str | None,
) -> None:
    require_project_venv()

    dst_dir = Path(output_dir) if output_dir else Path(default_output)
    staging = PYINSTALLER_BUILD_DIR / mode / "dist" / "ModularTranslator"
    internal_dir = dst_dir / "_internal"

    if not APP_ENTRY.exists():
        sys.exit(f"[错误] 找不到入口 {APP_ENTRY}")

    step(f"0/8 清理旧产物（{mode} 模式 -> {dst_dir}，保留 characters/ 与 dependencies/ 资产）")
    if dst_dir.exists():
        try:
            clean_output_dir(dst_dir)
            print("  已清理旧产物（characters/ 与 dependencies/ 资产已保留）")
        except Exception as exc:
            print(f"  [警告] 旧产物暂时无法清理（{exc}），将在 PyInstaller 构建后重试")

    engine_zip = prepare_engine(embed_engine)

    site_packages = find_site_packages(extra_paths)
    if not site_packages:
        sys.exit("[错误] 未找到项目 .venv 的 site-packages，构建中止")
    check_key_package_consistency(site_packages)
    vendor_imports = scan_vendor_imports(site_packages)
    print(f"  [vendor 扫描] 共 {len(vendor_imports)} 个依赖将加入 hidden imports")
    source_closure = expand_source_closure(site_packages, SOURCE_SEEDS)
    print(f"  [源码化闭包] {', '.join(source_closure)}")

    step("2/8 执行 PyInstaller 构建")
    cmd = build_pyinstaller_cmd(mode, engine_zip, site_packages, vendor_imports, source_closure, extra_paths)
    print("  " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        sys.exit(f"[错误] PyInstaller 构建失败，退出码 {r.returncode}")

    if not (staging / "ModularTranslator.exe").exists():
        sys.exit(f"[错误] 未找到 PyInstaller 产物 {staging / 'ModularTranslator.exe'}")

    step("2.5/8 将 PyInstaller 产物移动到目标目录")
    dst_dir.parent.mkdir(parents=True, exist_ok=True)
    if dst_dir.exists():
        # step 0 清理失败（目录被占用）时的兜底：清理但保留资产，再合并 staging
        clean_output_dir(dst_dir)
        for item in staging.iterdir():
            shutil.move(str(item), str(dst_dir))
        print(f"  {staging}/* -> {dst_dir}（合并，资产已保留）")
    else:
        shutil.move(str(staging), str(dst_dir))
        print(f"  {staging} -> {dst_dir}")

    if not (dst_dir / "ModularTranslator.exe").exists():
        sys.exit(f"[错误] 未找到产物 {dst_dir / 'ModularTranslator.exe'}")

    step("3/8 复刻运行期目录结构（仅空目录，不拷贝文件）")
    create_structure_dirs(ROOT, dst_dir, STRUCTURE_DIRS)

    copy_configs(dst_dir)
    copy_ffmpeg(dst_dir)
    copy_llama_release(mode, dst_dir)
    copy_torch_runtime(mode, include_cpu_fallback, dst_dir)
    copy_whisper_cuda(with_whisper_cuda, dst_dir)
    copy_characters_if_requested(with_characters, dst_dir)
    write_launchers(dst_dir)
    size_report(engine_zip, mode, dst_dir)

    if verify and not verify_package(mode, with_whisper_cuda, dst_dir, source_closure):
        sys.exit("[错误] 产物校验未通过，构建中止。")
    if smoke:
        smoke_test(dst_dir)

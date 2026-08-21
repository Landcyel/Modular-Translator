"""模型/角色资产分卷打包脚本 — 生成每个分卷不超过 1GB 的 7z 压缩包。

设计目标：
- 模型文件不进入主程序包，由本脚本单独制作分卷；
- 压缩包内保留 `dependencies/models/...` 与 `characters/...` 相对路径；
- 用户解压到 `D:\\ModularTranslator\\` 后目录结构即对齐。

用法：
  python tools/make_model_volumes.py
  python tools/make_model_volumes.py --only moss
  python tools/make_model_volumes.py --only gsv
  python tools/make_model_volumes.py --only translate
  python tools/make_model_volumes.py --volume-size 1g --output-dir release/models
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 需要打包进模型分卷的路径（相对项目根）
# 路径会原样保留在压缩包中，因此解压到产物根即可对齐目录结构。
MANIFEST = {
    "moss": [
        "dependencies/models/moss",
    ],
    "gsv": [
        "dependencies/models/v4",
        "dependencies/models/gsv",
    ],
    "translate": [
        "dependencies/models/sakura",
    ],
    "characters": [
        "characters",
    ],
}

# 默认全量打包顺序
DEFAULT_GROUPS = ["moss", "gsv", "translate", "characters"]


def find_7z() -> str | None:
    """定位 7-Zip 可执行文件。"""
    candidates = [
        shutil.which("7z"),
        shutil.which("7za"),
        r"C:\Program Files\7-Zip\7z.exe",
        r"C:\Program Files (x86)\7-Zip\7z.exe",
    ]
    for c in candidates:
        if c and Path(c).exists():
            return c
    return None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_archive(
    seven_zip: str,
    output_dir: Path,
    archive_base: Path,
    group: str,
    paths: list[str],
    volume_size: str,
) -> list[Path]:
    """用 7z 打包指定路径，返回生成的分卷文件列表。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / f"{archive_base.name}.7z"

    # 删除旧的同名分卷，避免残留
    for old in output_dir.glob(f"{archive_base.name}.7z.*"):
        old.unlink()

    cmd = [
        seven_zip, "a", "-t7z", "-mx=5", "-m0=LZMA2", "-mmt=on",
        f"-v{volume_size}",
        str(archive_path),
    ]
    added = 0
    for rel in paths:
        src = ROOT / rel
        if not src.exists():
            print(f"  [警告] 路径不存在，跳过: {src}")
            continue
        # 在 ROOT 下执行，确保压缩包内路径从 dependencies/models/... 开始
        cmd.append(rel)
        added += 1

    if added == 0:
        print(f"  [跳过] {group} 没有可打包路径")
        return []

    print(f"  [7z] {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        sys.exit(f"[错误] 7z 打包失败: {group}，退出码 {r.returncode}")

    volumes = sorted(output_dir.glob(f"{archive_base.name}.7z.*"))
    if not volumes:
        # 单卷时 7z 可能不生成 .7z.001，而是直接生成 .7z
        if archive_path.exists():
            volumes = [archive_path]
    return volumes


def write_checksum(output_dir: Path, volumes: list[Path], archive_base: Path) -> None:
    checksum_path = output_dir / f"{archive_base.name}.sha256"
    lines = []
    for v in volumes:
        lines.append(f"{sha256_file(v)}  {v.name}")
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  [校验] {checksum_path}")


def write_readme(output_dir: Path) -> None:
    readme = output_dir / "README.txt"
    readme.write_text(
        "模型分卷解压说明\n"
        "================\n"
        "1. 先解压主程序包到 D:\\ModularTranslator\n"
        "2. 将所有 .7z.001/.7z.002/... 放到同一目录\n"
        "3. 用 7-Zip 解压第一个分卷（如 models.part01.7z.001）到 D:\\ModularTranslator\n"
        "4. 解压后应出现：\n"
        "   dependencies\\models\\...\n"
        "   characters\\...\n"
        "5. 可用 models.sha256 校验分卷完整性\n",
        encoding="utf-8",
    )
    print(f"  [说明] {readme}")


def main() -> None:
    ap = argparse.ArgumentParser(description="模型/角色资产 1GB 分卷打包")
    ap.add_argument(
        "--only",
        nargs="+",
        choices=list(MANIFEST.keys()),
        help="只打包指定分组，例如 --only moss gsv",
    )
    ap.add_argument(
        "--volume-size",
        default="1g",
        help="分卷大小，默认 1g（7z 语法，如 1g/1024m）",
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "release" / "models"),
        help="输出目录，默认 release/models",
    )
    ap.add_argument(
        "--archive-name",
        default="models",
        help="压缩包基础名，默认 models",
    )
    ap.add_argument(
        "--no-characters",
        action="store_true",
        help="不打包 characters/ 角色资产（默认全量打包时排除该分组）",
    )
    args = ap.parse_args()

    seven_zip = find_7z()
    if seven_zip is None:
        sys.exit(
            "[错误] 未找到 7-Zip。\n"
            "  请安装 7-Zip 或将其加入 PATH，例如：\n"
            "    C:\\Program Files\\7-Zip\\7z.exe"
        )

    groups = list(args.only) if args.only else list(DEFAULT_GROUPS)
    if args.no_characters:
        groups = [g for g in groups if g != "characters"]
        print("  [排除] 已排除 characters/ 角色资产")
    output_dir = Path(args.output_dir).resolve()
    archive_base = Path(args.archive_name)

    all_volumes: list[Path] = []
    for group in groups:
        if group not in MANIFEST:
            print(f"  [跳过] 未知分组: {group}")
            continue
        print(f"== 打包分组: {group} ==")
        group_base = archive_base if len(groups) == 1 else archive_base.with_name(f"{archive_base.name}-{group}")
        volumes = build_archive(
            seven_zip=seven_zip,
            output_dir=output_dir,
            archive_base=group_base,
            group=group,
            paths=MANIFEST[group],
            volume_size=args.volume_size,
        )
        all_volumes.extend(volumes)

    if all_volumes:
        write_checksum(output_dir, all_volumes, archive_base)
    write_readme(output_dir)
    print(f"\n完成，输出目录: {output_dir}")


if __name__ == "__main__":
    main()

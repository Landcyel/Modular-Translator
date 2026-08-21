"""双版本物理副本刷新脚本 — 把 git 分支最新内容重新导出到 app_workflow/ 与 app_lite/。

背景：
- 项目采用「git 分支（权威版本源）+ 物理副本（可独立运行/打包的快照）」双轨。
  git 分支 app_workflow / app_lite 各自保存两个版本的代码；
  根目录 app_workflow/ 与 app_lite/ 是它们的工作副本（.gitignore 排除），
  其中 dependencies/、build_assets/、tests/data/ 以 junction 指向根共享资源。
- 分支代码更新后，重跑本脚本即可把副本刷新到分支最新状态（幂等）。

用法：
  python tools/sync_version_dirs.py            # 刷新两个版本
  python tools/sync_version_dirs.py --lite     # 仅刷新 app_lite
  python tools/sync_version_dirs.py --dry-run  # 仅打印将执行的操作

安全说明：
- 只会删除/重建 app_workflow/ 与 app_lite/ 两个副本目录（含其内 junction，
  junction 删除仅移除联接本身，不触碰共享资源实体）。
- 共享资源目标缺失时跳过 junction 创建并打印警告，不失败。
"""

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 版本目录名 → 来源 git 分支名（保持一致）
VERSIONS = {"app_workflow": "app_workflow", "app_lite": "app_lite"}

# 副本内以 junction 挂载的共享资源（相对项目根）；目标缺失时跳过
# characters/ 为 GSV 角色资产（S1/S2 权重 + 参考音频，gitignore），副本必须共享
SHARED_RELS = ["dependencies", "build_assets", "tests/data", "characters"]


def _run_git(args: list[str]) -> str:
    r = subprocess.run(["git", *args], cwd=str(ROOT), capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"[错误] git {' '.join(args)} 失败: {r.stderr.strip()}")
    return r.stdout.strip()


def _remove_tree_keep_root(dst: Path) -> None:
    """删除副本目录内容（junction 用 rmdir 只移除联接，普通目录/文件递归删除）。"""
    for child in dst.iterdir():
        if child.is_dir() and child.is_junction():
            child.rmdir()          # junction：rmdir 只移除联接本身，不触碰目标
        elif child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _make_junction(link: Path, target: Path) -> bool:
    """创建目录 junction（Windows mklink /J）。目标不存在返回 False。"""
    if not target.is_dir():
        return False
    link.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
    )
    return r.returncode == 0


def sync_version(dirname: str, branch: str, dry_run: bool) -> None:
    dst = ROOT / dirname
    if dry_run:
        print(f"[dry-run] 将删除并重建 {dirname}/（来源分支: {branch}）")
        return
    print(f"== 刷新 {dirname}/ ← 分支 {branch} ==")
    if dst.exists():
        _remove_tree_keep_root(dst)
    else:
        dst.mkdir(parents=True)

    # git archive 导出跟踪文件（不含 dependencies/ 等 gitignore 大资源）
    tmp_tar = Path(tempfile.mkdtemp()) / f"{branch}.tar"
    _run_git(["archive", branch, "-o", str(tmp_tar)])
    with tarfile.open(tmp_tar, "r") as tf:
        tf.extractall(dst, filter="data")
    tmp_tar.unlink()
    print(f"  已导出 {branch} 跟踪文件 → {dst}")

    # junction 挂载共享资源
    for rel in SHARED_RELS:
        link = dst / rel
        target = ROOT / rel
        if _make_junction(link, target):
            print(f"  junction: {rel}/ -> {target}")
        else:
            print(f"  [警告] 共享资源缺失，跳过 junction: {target}")
    print(f"  完成: {dst}")


def main() -> None:
    ap = argparse.ArgumentParser(description="刷新双版本物理副本（app_workflow/app_lite）")
    ap.add_argument("--lite", action="store_true", help="仅刷新 app_lite")
    ap.add_argument("--dry-run", action="store_true", help="仅打印操作，不执行")
    args = ap.parse_args()

    targets = {"app_lite": "app_lite"} if args.lite else VERSIONS
    for dirname, branch in targets.items():
        sync_version(dirname, branch, args.dry_run)


if __name__ == "__main__":
    main()

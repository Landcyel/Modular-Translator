"""nltk 数据安装脚本（GPT-SoVITS 英文 G2P 依赖）。

vendored text/english.py 与 g2p_en 需要:
- corpora/cmudict                       （g2p_en G2p.__init__ 用）
- corpora/cmudict.zip                   （g2p_en 用 find('corpora/cmudict.zip') 探测）
- taggers/averaged_perceptron_tagger_eng（nltk.pos_tag 用, nltk>=3.9 新格式）
- taggers/averaged_perceptron_tagger.zip（g2p_en 用 find('...tagger.zip') 探测）

上游数据源 raw.githubusercontent.com 在本机被 DNS 污染（解析到 127.0.0.1，
nltk 的 pathsec 会拦截并报 SSRF），因此按序尝试 CDN/代理镜像拉取。

用法: .venv/Scripts/python -m core.gsv.setup_nltk_data
"""

from __future__ import annotations

import io
import os
import sys
import zipfile
from pathlib import Path

import nltk  # 确保 nltk 已安装

RAW_BASE = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages"
# 按序尝试: jsdelivr CDN → 通用 GitHub 代理
SOURCES = [
    f"https://cdn.jsdelivr.net/gh/nltk/nltk_data@gh-pages/packages",
    f"https://ghproxy.net/{RAW_BASE}",
    f"https://gh-proxy.com/{RAW_BASE}",
]

# (资源名, 相对 nltk_data 的落点, 是否解压)
JOBS = [
    ("corpora/cmudict", "corpora", True),
    ("corpora/cmudict.zip", "corpora", False),
    ("taggers/averaged_perceptron_tagger.zip", "taggers", False),
    ("taggers/averaged_perceptron_tagger_eng", "taggers", True),
]


def nltk_data_root() -> Path:
    # nltk 搜索路径之一: <venv>/nltk_data（gitignored, 项目作用域）
    return Path(sys.prefix) / "nltk_data"


def main() -> int:
    import urllib.error
    import urllib.request

    root = nltk_data_root()
    for res, category, unzip in JOBS:
        dest = root / category
        dest.mkdir(parents=True, exist_ok=True)
        target = dest / Path(res).name
        if unzip:
            if target.is_dir() and any(target.iterdir()):
                print(f"  [skip] {res} (exists)")
                continue
        elif target.exists() and target.stat().st_size > 0:
            print(f"  [skip] {res} (exists)")
            continue
        suffix = "" if res.endswith(".zip") else ".zip"
        data = None
        for base in SOURCES:
            url = f"{base}/{res}{suffix}"
            try:
                print(f"  [get ] {res} <- {url}", end=" ... ", flush=True)
                data = urllib.request.urlopen(url, timeout=180).read()
                print("OK", end=" ")
                break
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
                print(f"fail({e.__class__.__name__})", end=" ")
        if data is None:
            print(f"!! 所有镜像均失败: {res}")
            return 1
        if unzip:
            zipfile.ZipFile(io.BytesIO(data)).extractall(dest)
            print(f"({len(data)/1e6:.1f}MB, unzipped to {target})")
        else:
            target.write_bytes(data)
            print(f"({len(data)/1e6:.1f}MB)")

    # 验证
    from nltk.corpus import cmudict
    from nltk import pos_tag

    print(f"  [ok] cmudict words: {len(cmudict.dict())}")
    print(f"  [ok] pos_tag: {pos_tag(['hello'])}")
    # g2p_en 的 zip 探测（无网络尝试即通过）
    import g2p_en  # noqa: F401

    nltk.data.find("corpora/cmudict.zip")
    nltk.data.find("taggers/averaged_perceptron_tagger.zip")
    print("  [ok] g2p_en zip probes pass (无下载尝试)")
    print("NLTK_DATA_READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

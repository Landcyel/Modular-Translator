"""nltk data installer script (GPT-SoVITS English G2P dependency).

The vendored text/english.py and g2p_en require:
- corpora/cmudict                       (used by g2p_en G2p.__init__)
- corpora/cmudict.zip                   (probed by g2p_en via find('corpora/cmudict.zip'))
- taggers/averaged_perceptron_tagger_eng (used by nltk.pos_tag; new format for nltk>=3.9)
- taggers/averaged_perceptron_tagger.zip (probed by g2p_en via find('...tagger.zip'))

The upstream data source raw.githubusercontent.com is DNS-polluted on this machine
(resolves to 127.0.0.1, and nltk's pathsec blocks it as SSRF), so CDN/proxy mirrors
are tried in order.

Usage: .venv/Scripts/python -m core.gsv.setup_nltk_data
"""

from __future__ import annotations

import io
import os
import sys
import zipfile
from pathlib import Path

import nltk  # ensure nltk is installed

RAW_BASE = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages"
# Try in order: jsdelivr CDN → generic GitHub proxy
SOURCES = [
    f"https://cdn.jsdelivr.net/gh/nltk/nltk_data@gh-pages/packages",
    f"https://ghproxy.net/{RAW_BASE}",
    f"https://gh-proxy.com/{RAW_BASE}",
]

# (resource name, destination relative to nltk_data, whether to unzip)
JOBS = [
    ("corpora/cmudict", "corpora", True),
    ("corpora/cmudict.zip", "corpora", False),
    ("taggers/averaged_perceptron_tagger.zip", "taggers", False),
    ("taggers/averaged_perceptron_tagger_eng", "taggers", True),
]


def nltk_data_root() -> Path:
    # One of nltk's search paths: <venv>/nltk_data (gitignored, project-scoped)
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

    # Verify
    from nltk.corpus import cmudict
    from nltk import pos_tag

    print(f"  [ok] cmudict words: {len(cmudict.dict())}")
    print(f"  [ok] pos_tag: {pos_tag(['hello'])}")
    # g2p_en's zip probes (pass without network attempts)
    import g2p_en  # noqa: F401

    nltk.data.find("corpora/cmudict.zip")
    nltk.data.find("taggers/averaged_perceptron_tagger.zip")
    print("  [ok] g2p_en zip probes pass (无下载尝试)")
    print("NLTK_DATA_READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

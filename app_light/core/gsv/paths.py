"""模型权重路径解析: dependencies/models/ 相对项目根 + GSV_* 环境变量覆盖。

默认（v2ProPlus 主线）:
- s1/t2s : dependencies/models/v4/s1v3.ckpt
- s2/vits: dependencies/models/gsv/v2proplus/s2Gv2ProPlus.pth
- bert   : dependencies/models/v4/chinese-roberta-wwm-ext-large
- hubert : dependencies/models/v4/chinese-hubert-base
- sv     : dependencies/models/gsv/sv/pretrained_eres2netv2w24s4ep4.ckpt
- g2pw   : dependencies/models/gsv/g2pw/G2PWModel
- langdetect: dependencies/models/gsv/fast_langdetect
- vocoder: dependencies/models/v4/gsv-v4-pretrained （v4 用）

环境变量覆盖（优先级高于 config dict 默认，低于 config dict 显式传值）:
GSV_T2S_PATH / GSV_VITS_PATH / GSV_BERT_PATH / GSV_HUBERT_PATH /
GSV_SV_PATH / GSV_G2PW_DIR / GSV_LANGDETECT_DIR / GSV_VOCODER_DIR
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_ROOT = PROJECT_ROOT / "dependencies" / "models"

# 各版本默认 s2（s1 全部共用 s1v3.ckpt）
DEFAULT_VITS = {
    "v1": "dependencies/models/v4/gsv-v4-pretrained/s2Gv4.pth",  # 占位（本仓库未备 v1/v2/v3 权重）
    "v2": "dependencies/models/v4/gsv-v4-pretrained/s2Gv4.pth",
    "v2Pro": "dependencies/models/gsv/v2proplus/s2Gv2ProPlus.pth",
    "v2ProPlus": "dependencies/models/gsv/v2proplus/s2Gv2ProPlus.pth",
    "v3": "dependencies/models/v4/gsv-v4-pretrained/s2Gv4.pth",
    "v4": "dependencies/models/v4/gsv-v4-pretrained/s2Gv4.pth",
}

VALID_VERSIONS = {"v1", "v2", "v3", "v4", "v2Pro", "v2ProPlus"}


def _abs(p: str | Path) -> str:
    p = Path(p)
    return str(p if p.is_absolute() else (PROJECT_ROOT / p))


def _env_or(key: str, default: Path) -> Path:
    v = os.environ.get(key)
    return Path(v) if v else default


def resolve_config(config: Optional[dict] = None) -> dict:
    """归一化引擎配置: 全部路径转绝对路径，补全默认值。

    返回 dict 键:
    version / device / is_half / t2s_weights_path / vits_weights_path /
    bert_base_path / cnhuhbert_base_path / sv_path / sv_dir / g2pw_dir /
    langdetect_dir / vocoder_dir
    """
    config = dict(config or {})
    version = config.get("version", "v2ProPlus")
    if version not in VALID_VERSIONS:
        raise ValueError(f"Invalid GSV version: {version!r}")

    device = config.get("device", "auto")
    if device == "auto":
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    is_half = config.get("is_half", device.startswith("cuda"))

    bert = _env_or("GSV_BERT_PATH", MODELS_ROOT / "v4/chinese-roberta-wwm-ext-large")
    hubert = _env_or("GSV_HUBERT_PATH", MODELS_ROOT / "v4/chinese-hubert-base")
    sv_ckpt = _env_or(
        "GSV_SV_PATH", MODELS_ROOT / "gsv/sv/pretrained_eres2netv2w24s4ep4.ckpt"
    )
    g2pw = _env_or("GSV_G2PW_DIR", MODELS_ROOT / "gsv/g2pw/G2PWModel")
    langdetect = _env_or("GSV_LANGDETECT_DIR", MODELS_ROOT / "gsv/fast_langdetect")
    vocoder = _env_or(
        "GSV_VOCODER_DIR", MODELS_ROOT / "v4/gsv-v4-pretrained"
    )

    def pick(key: str, default: Path) -> str:
        v = config.get(key)
        return _abs(v) if v else _abs(_env_or("", default))

    return {
        "version": version,
        "device": device,
        "is_half": bool(is_half),
        "t2s_weights_path": pick("t2s_weights_path", MODELS_ROOT / "v4/s1v3.ckpt"),
        "vits_weights_path": pick("vits_weights_path", Path(DEFAULT_VITS[version])),
        "bert_base_path": pick("bert_base_path", bert),
        "cnhuhbert_base_path": pick("cnhuhbert_base_path", hubert),
        "sv_path": pick("sv_path", sv_ckpt),
        "sv_dir": str(sv_ckpt.parent),
        "g2pw_dir": str(g2pw),
        "langdetect_dir": str(langdetect),
        "vocoder_dir": str(vocoder),
    }


def merge_service_role(service_cfg: Optional[dict], role_cfg: Optional[dict]) -> dict:
    """合并 GSV 服务配置（models/gsv/default.json）与角色配置（tts/roles/role-*.json）。

    角色配置覆盖服务配置的同名键（S1/S2 权重仅角色配置携带，服务配置
    不含 t2s/vits 键，避免 weights_status() 的"角色显式指定"语义被默认值污染）。
    无角色配置时仅返回服务配置（引擎用默认权重加载）。
    """
    merged = dict(service_cfg or {})
    if role_cfg:
        merged.update(role_cfg)
    return merged

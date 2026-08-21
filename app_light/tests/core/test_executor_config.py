"""回归测试：任务提交配置生效性 + 未选（None）友好报错。

覆盖：
- 选择生效：task.configs 里所选 prompts/args 路径被解析为 dict 并传入 _translate
- prompts=None → ValueError 友好消息（替代底层 TypeError）
- translate_config=None → ValueError 友好消息
"""
from pathlib import Path

import pytest

from core.contracts import Task
from core.executor import Translator

ROOT = Path(__file__).resolve().parents[2]

ARGS = ROOT / "configs/translate/args_llama/default.json"
PROMPTS = ROOT / "configs/translate/prompts/default.json"


class _T(Translator):
    """mock _translate：捕获收到的配置并返回原文（走真实 _resolve_task/translate 流程）。"""

    def __init__(self, config=None):
        super().__init__(config)
        self.captured = []

    def _translate(self, text, trans_config, prompts, glossary=None,
                   timeout=None, cancel_event=None, pause_event=None):
        self.captured.append((text, trans_config, prompts, glossary))
        return text


def _task(src: Path, configs: dict) -> Task:
    return Task(task_type="llama", file_path=src, file_name="x.txt", configs=configs)


def test_selected_configs_are_used(tmp_path):
    """选择生效：所选配置路径解析为 dict 并实际传入 _translate。"""
    src = tmp_path / "src.txt"
    src.write_text("こんにちは\n世界\n", encoding="utf-8")
    t = _T()
    t.execute(_task(src, {"translate_config": ARGS, "prompts": PROMPTS}))
    assert t.captured, "应有 _translate 调用"
    _, trans_config, prompts, _ = t.captured[0]
    assert isinstance(trans_config, dict) and "request" in trans_config
    assert isinstance(prompts, dict) and "system" in prompts


def test_prompts_none_raises_friendly_error(tmp_path):
    """prompts 未选（None）→ 显式 ValueError（友好消息），而非底层 TypeError。"""
    src = tmp_path / "src.txt"
    src.write_text("こんにちは\n", encoding="utf-8")
    t = _T()
    with pytest.raises(ValueError, match="提示词配置未选择"):
        t.execute(_task(src, {"translate_config": ARGS, "prompts": None}))


def test_translate_config_none_raises_friendly_error(tmp_path):
    """translate_config 未选（None）→ 显式 ValueError（友好消息）。"""
    src = tmp_path / "src.txt"
    src.write_text("こんにちは\n", encoding="utf-8")
    t = _T()
    with pytest.raises(ValueError, match="翻译参数配置未选择"):
        t.execute(_task(src, {"translate_config": None, "prompts": PROMPTS}))

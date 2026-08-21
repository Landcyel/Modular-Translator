"""回归测试：start_service 使所选配置生效（服务配置选择不生效修复）。

覆盖：
- start_service(name, None, config_path)：service 用所选配置构造，_config_dict 同步
- 不传 config_path：回退注册表默认（现有行为不变）
- restart 分支：已存在停止中的 service 用传入新配置重启
- LlamaService 空 llama_path：归一化为 None → 启动抛 "llama_path not configured"
"""
from pathlib import Path

import pytest

from core.facade import Facade, Service
from core.service import LlamaService
from core.task_que import TranslationTaskQueue

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CFG = Path("configs/models/llama/default.json")


class _DummyService(Service):
    """记录构造/重启收到的配置路径。"""

    def __init__(self, model_config, on_status_change=None):
        super().__init__(model_config, on_status_change)
        self.config = model_config

    def start(self):
        self._running = True

    def stop(self):
        self._running = False

    def restart(self, config):
        self.config = config

    def get_executor(self):
        return None


def _facade() -> Facade:
    return Facade(
        backend_dict={"llama": (_DummyService, TranslationTaskQueue)},
        config_dict={"llama": DEFAULT_CFG},
    )


def test_start_service_uses_selected_config(tmp_path):
    """传 config_path → service 用所选配置构造，且 _config_dict 同步（后续 restart 沿用）。"""
    selected = tmp_path / "llama_selected.json"
    selected.write_text("{}", encoding="utf-8")
    f = _facade()
    f.start_service("llama", None, selected)
    assert f._service_dic["llama"].config == selected
    assert f._config_dict["llama"] == selected


def test_start_service_falls_back_to_registry():
    """不传 config_path → 回退注册表默认配置。"""
    f = _facade()
    f.start_service("llama", None, None)
    assert f._service_dic["llama"].config == DEFAULT_CFG


def test_start_service_restart_applies_new_config(tmp_path):
    """已存在停止中的 service → restart 分支使用传入的新配置。"""
    selected = tmp_path / "llama_selected.json"
    selected.write_text("{}", encoding="utf-8")
    f = _facade()
    f.start_service("llama", None, selected)
    f.stop_service("llama")
    f.start_service("llama", None, DEFAULT_CFG)
    assert f._service_dic["llama"].config == DEFAULT_CFG
    assert f._config_dict["llama"] == DEFAULT_CFG


def test_llama_empty_path_not_configured():
    """空 llama_path 归一化为 None → 启动抛 llama_path not configured。"""
    svc = LlamaService({"llama_path": ""})
    assert svc._llama_path is None
    with pytest.raises(RuntimeError, match="llama_path not configured"):
        svc._start_llama_server()


def test_llama_nonempty_path_resolved():
    """非空 llama_path 解析为项目根下的绝对路径。"""
    svc = LlamaService({"llama_path": "dependencies/llama-release"})
    assert svc._llama_path is not None
    assert svc._llama_path.name == "llama-release"


def test_submit_task_logs_task_configs_paths():
    """提交任务日志显示任务实际使用的配置路径（翻译参数/提示词），而非服务配置。"""
    from app.facade import AppFacade
    from app.log import log
    from core.contracts import TranslationRequest

    ARGS = ROOT / "configs/translate/args_llama/default.json"
    PROMPTS = ROOT / "configs/translate/prompts/default.json"
    facade = AppFacade(
        backend_dict={"llama": (_DummyService, TranslationTaskQueue)},
        config_dict={"llama": DEFAULT_CFG},
    )
    log.clear()
    req = TranslationRequest(
        task_type="llama",
        file_path=Path("a.txt"),
        file_name="a.txt",
        configs={"translate_config": ARGS, "prompts": PROMPTS},
    )
    facade.submit_task(req)
    joined = "\n".join(log.lines())
    assert "翻译参数: " in joined and "args_llama" in joined, joined
    assert "提示词: " in joined and "prompts" in joined, joined
    # 不含服务配置路径（configs/models/...）
    assert "models" not in joined, joined

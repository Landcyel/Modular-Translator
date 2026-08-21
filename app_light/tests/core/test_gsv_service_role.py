"""GSV 角色热切换（S1/S2 与基础模型分离加载）测试。

覆盖三层：
- GsvEngine.apply_role：S1/S2 热切换调用序列、CPU dtype 归一、参考缓存失效、
  角色状态记录、released 保护
- GsvService.switch_role：运行中热切换 vs 未运行全量重启
- AppFacade.switch_service_config：gsv 运行中走热切换分支（取消当前任务）
  vs 未运行保持 stop+start

引擎/服务均为 fake 对象，不触发真实模型加载。
"""

import os
import threading
from pathlib import Path

import pytest


# ── fakes ─────────────────────────────────────────────────────

class _FakeModule:
    """可 .float() 的假 nn.Module 替身（记录是否被归一）。"""

    def __init__(self):
        self.floated = False

    def float(self):
        self.floated = True


class _FakeTTS:
    """记录 init_t2s_weights / init_vits_weights / set_ref_audio 调用。"""

    def __init__(self):
        self.calls = []
        self.t2s_model = _FakeModule()
        self.vits_model = _FakeModule()
        self.prompt_cache = {"ref_audio_path": "old_ref.wav"}

    def init_t2s_weights(self, path):
        self.calls.append(("t2s", str(path)))

    def init_vits_weights(self, path):
        self.calls.append(("vits", str(path)))

    def set_ref_audio(self, path):
        self.calls.append(("ref", str(path)))
        self.prompt_cache["ref_audio_path"] = str(path)


class _FakeEngine:
    """记录 apply_role 调用的假 GsvEngine。"""

    def __init__(self, released=False):
        self.apply_calls = []
        self._released = released

    def apply_role(self, cfg):
        self.apply_calls.append(cfg)

    def weights_status(self):
        return {
            "t2s_specified": False, "t2s_role_loaded": True,
            "t2s_configured": "", "t2s_used": "s1.ckpt",
            "vits_specified": False, "vits_role_loaded": True,
            "vits_configured": "", "vits_used": "s2.pth",
        }


class _FakeGsvService:
    """facade 分支用的假 GsvService（is_running + switch_role）。"""

    def __init__(self, running=False):
        self.is_running = running
        self.switch_calls = []

    def switch_role(self, cfg):
        self.switch_calls.append(cfg)


class _TaskStub:
    """facade 热切换分支取消的当前任务替身（属性访问 id）。"""
    id = "t1"


class _FakeQueue:
    def __init__(self):
        self.cancelled = []

    def get_current_task(self):
        return _TaskStub()

    def cancel(self, tid):
        self.cancelled.append(tid)


def _make_engine(tts=None, device="cpu", released=False):
    """构造未跑 __init__ 的 GsvEngine（避免真实模型加载）。"""
    from core.gsv import engine as engine_mod

    eng = engine_mod.GsvEngine.__new__(engine_mod.GsvEngine)
    eng._lock = threading.RLock()
    eng._vendor_root = engine_mod.VENDOR_ROOT
    eng._tts = tts or _FakeTTS()
    eng._released = released
    eng._cfg = {
        "device": device,
        "t2s_weights_path": "old_t2s.ckpt",
        "vits_weights_path": "old_vits.pth",
    }
    eng._raw_config = {}
    return eng


# ── 1. GsvEngine.apply_role ───────────────────────────────────

def test_apply_role_hot_swaps_s1_s2_in_order():
    """热切换按 S1 → S2 顺序重建，路径绝对化。"""
    eng = _make_engine()
    eng.apply_role({
        "t2s_weights_path": "characters/ookura_lumine[v2ProPlus]/ookura_lumine-e30.ckpt",
        "vits_weights_path": "characters/ookura_lumine[v2ProPlus]/ookura_lumine_e8_s520.pth",
        "role_ref_audio": "characters/ookura_lumine[v2ProPlus]/ref.wav",
        "prompt_text": "台词",
    })
    kinds = [c[0] for c in eng._tts.calls]
    assert kinds == ["t2s", "vits", "ref"], f"调用顺序异常: {kinds}"
    # 路径绝对化（相对 CWD）
    assert eng._tts.calls[0][1] == os.path.abspath(
        "characters/ookura_lumine[v2ProPlus]/ookura_lumine-e30.ckpt")
    # 角色状态记录（weights_status 的 specified/configured 语义）
    assert eng._raw_config["t2s_weights_path"].endswith("ookura_lumine-e30.ckpt")
    assert eng._cfg["t2s_weights_path"].endswith("ookura_lumine-e30.ckpt")
    assert eng._cfg["vits_weights_path"].endswith("ookura_lumine_e8_s520.pth")


def test_apply_role_cpu_normalizes_swapped_modules():
    """CPU 模式下热切换替换出的新 S1/S2 补 .float()。"""
    eng = _make_engine(device="cpu")
    eng.apply_role({
        "t2s_weights_path": "characters/白[v2Proplus]/shiro-e15.ckpt",
        "vits_weights_path": "characters/白[v2Proplus]/shiro_e8_s144.pth",
    })
    assert eng._tts.t2s_model.floated is True
    assert eng._tts.vits_model.floated is True


def test_apply_role_cuda_skips_float():
    """CUDA 模式不执行 float 归一（由 is_half 各自处理）。"""
    eng = _make_engine(device="cuda")
    eng.apply_role({"t2s_weights_path": "a.ckpt", "vits_weights_path": "b.pth"})
    assert eng._tts.t2s_model.floated is False
    assert eng._tts.vits_model.floated is False


def test_apply_role_ref_audio_recomputes_cache():
    """角色配置带 role_ref_audio → set_ref_audio 重算预热缓存。"""
    eng = _make_engine()
    eng.apply_role({"t2s_weights_path": "a.ckpt", "vits_weights_path": "b.pth",
                    "role_ref_audio": "refs/r1.wav"})
    assert eng._tts.prompt_cache["ref_audio_path"] == os.path.abspath("refs/r1.wav")


def test_apply_role_without_ref_audio_invalidates_cache():
    """无参考音频 → 置 ref_audio_path=None 使旧缓存失效（run 强制重算）。"""
    eng = _make_engine()
    eng.apply_role({"t2s_weights_path": "a.ckpt", "vits_weights_path": "b.pth"})
    assert eng._tts.prompt_cache["ref_audio_path"] is None


def test_apply_role_uses_existing_weights_when_missing():
    """角色配置缺 S1/S2 键时沿用现役权重路径。"""
    eng = _make_engine()
    eng.apply_role({"prompt_text": "仅换文本"})
    assert eng._tts.calls[0][1] == os.path.abspath("old_t2s.ckpt")
    assert eng._tts.calls[1][1] == os.path.abspath("old_vits.pth")


def test_apply_role_released_raises():
    """引擎已释放时抛 RuntimeError。"""
    eng = _make_engine(released=True)
    with pytest.raises(RuntimeError):
        eng.apply_role({"t2s_weights_path": "a.ckpt", "vits_weights_path": "b.pth"})


# ── 2. GsvService.switch_role ─────────────────────────────────

def _make_service():
    from core.service import GsvService

    svc = GsvService.__new__(GsvService)
    svc._engine = None
    svc._executor = None
    svc._running = False
    svc._config = {}
    svc._on_status_change = None
    svc._log = lambda *a, **k: None
    svc._emit = lambda: None
    svc._resolve_config({})   # 初始化 _config（相对路径绝对化）
    return svc


def test_switch_role_running_hot_swaps():
    """引擎运行中：apply_role 被调用，_config 更新为角色配置。"""
    svc = _make_service()
    svc._engine = _FakeEngine()
    svc._running = True
    svc.switch_role({
        "t2s_weights_path": "characters/ookura_lumine[v2ProPlus]/ookura_lumine-e30.ckpt",
        "vits_weights_path": "characters/ookura_lumine[v2ProPlus]/ookura_lumine_e8_s520.pth",
    })
    assert len(svc._engine.apply_calls) == 1
    assert svc._engine.apply_calls[0]["t2s_weights_path"].endswith("ookura_lumine-e30.ckpt")
    assert svc._config["t2s_weights_path"].endswith("ookura_lumine-e30.ckpt")


def test_switch_role_released_engine_full_restart(monkeypatch):
    """引擎已释放：回退全量重启（restart → start）。"""
    svc = _make_service()
    svc._engine = _FakeEngine(released=True)
    svc._running = False
    started = []

    def _fake_start(self):
        started.append(self._config)

    monkeypatch.setattr("core.service.GsvService.start", _fake_start)
    svc.switch_role({"t2s_weights_path": "a.ckpt", "vits_weights_path": "b.pth"})
    assert started, "未运行时应走全量 start"
    assert len(svc._engine.apply_calls) == 0


def test_switch_role_not_running_full_restart(monkeypatch):
    """服务未启动：回退全量重启。"""
    svc = _make_service()
    started = []

    def _fake_start(self):
        started.append(self._config)

    monkeypatch.setattr("core.service.GsvService.start", _fake_start)
    svc.switch_role({"t2s_weights_path": "a.ckpt", "vits_weights_path": "b.pth"})
    assert started, "未运行时走全量 start"


# ── 3. AppFacade.switch_service_config（gsv 分支）──────────────

def _make_facade(service=None, queue=None):
    from app.facade import AppFacade

    facade = AppFacade({}, {})
    if service is not None:
        facade._service_dic["gsv"] = service
    if queue is not None:
        facade._queue_dic["gsv"] = queue
    facade._stop_calls = []
    facade._start_calls = []
    facade._orig_stop = facade.stop_service
    facade._orig_start = facade.start_service

    def _fake_stop(name, cancel_current=False):
        facade._stop_calls.append((name, cancel_current))

    def _fake_start(name, backend=None, config_path=None):
        facade._start_calls.append((name, config_path))

    facade.stop_service = _fake_stop
    facade.start_service = _fake_start
    return facade


def test_switch_config_gsv_running_hot_swaps_and_cancels():
    """gsv 运行中：热切换 + 取消当前任务，不走 stop/start。"""
    svc = _FakeGsvService(running=True)
    queue = _FakeQueue()
    facade = _make_facade(svc, queue)
    facade.switch_service_config("gsv", {"t2s_weights_path": "a.ckpt",
                                         "vits_weights_path": "b.pth"})
    assert len(svc.switch_calls) == 1
    assert queue.cancelled == ["t1"], "热切换前应取消运行中任务"
    assert facade._stop_calls == [] and facade._start_calls == []


def test_switch_config_gsv_not_running_full_restart():
    """gsv 未运行：保持 stop+start 全量重启。"""
    svc = _FakeGsvService(running=False)
    facade = _make_facade(svc, _FakeQueue())
    facade.switch_service_config("gsv", {"t2s_weights_path": "a.ckpt"})
    assert len(svc.switch_calls) == 0
    assert len(facade._stop_calls) == 1 and facade._stop_calls[0][1] is True
    assert len(facade._start_calls) == 1


def test_switch_config_other_service_unchanged():
    """非 gsv 服务保持原 stop+start 语义。"""
    facade = _make_facade()
    facade.switch_service_config("llama", "cfg.json")
    assert len(facade._stop_calls) == 1
    assert len(facade._start_calls) == 1
    assert facade._start_calls[0] == ("llama", "cfg.json")

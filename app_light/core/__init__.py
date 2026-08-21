"""
Translator Core — public API.

惰性导出（PEP 562 ``__getattr__``）：``from core import X`` 仍兼容，但导入
单个子模块（如 ``core.contracts`` / ``core.system_config``）不再触发整包聚合
导入。聚合导入曾连带加载 openai / numpy 重链（实测 ≈2.4s），
拖慢应用骨架首帧；重模块统一延迟到 UI 后台初始化时导入。

All imports are via package-relative paths; no sys.path hacks needed.
"""

from app import torch_runtime  # noqa: F401  # 任何 core 入口都先完成可插拔 torch 运行时选择（轻量，不 import torch）

__all__ = [
    # task queue
    "Task", "TaskQueue", "TranslationTaskQueue", "TranscriptionTaskQueue", "TaskStatus",
    "GsvTaskQueue",
    "Backends",
    # translators
    "Translator", "LlamaTranslator", "APITranslator",
    # transcriber / executor
    "GsvTTSExecutor",
    # rule splitter
    "RuleSplitter", "SentenceInfo",
    # service
    "Service", "LlamaService", "APIService", "GsvService",
    # gsv engine
    "GsvEngine",
    # moss backend
    "MossService", "MossTranscriber", "MossSubprocessService", "MossHttpTranscriber",
    "force_single_speaker", "export_lrc",
    # contracts
    "TranslationRequest", "TranscriptionRequest",
    "TaskSnapshot",
    # facade
    "CoreFacade",
    # utils
    "load_json_file", "load_noval_file",
    "get_device_list",
    # writer
    "Segment", "lrc_write", "merge_repeated_segments",
]

# 惰性导出映射：名字 → 所在子模块名（相对 core 包）
_LAZY_EXPORTS = {
    # task queue（Task/TaskStatus 由 task_que 从 contracts 转发）
    "Task": "task_que",
    "TaskQueue": "task_que",
    "TranslationTaskQueue": "task_que",
    "TranscriptionTaskQueue": "task_que",
    "GsvTaskQueue": "task_que",
    "TaskStatus": "task_que",
    "Backends": "task_que",
    # executor
    "Translator": "executor",
    "LlamaTranslator": "executor",
    "APITranslator": "executor",
    "GsvTTSExecutor": "executor",
    "Executor": "executor",
    # rule splitter
    "RuleSplitter": "rule_splitter",
    "SentenceInfo": "rule_splitter",
    # service
    "Service": "service",
    "LlamaService": "service",
    "APIService": "service",
    "GsvService": "service",
    "GsvEngine": "gsv",
    "get_device_list": "service",
    # moss backend
    "MossService": "moss.moss_service",
    "MossTranscriber": "moss.moss_transcriber",
    "MossSubprocessService": "moss.moss_subprocess",
    "MossHttpTranscriber": "moss.moss_subprocess",
    "force_single_speaker": "moss.speaker_utils",
    "export_lrc": "moss.lrc_export",
    # contracts
    "TranslationRequest": "contracts",
    "TranscriptionRequest": "contracts",
    "TaskSnapshot": "contracts",
    # facade（子模块内实际属性名为 Facade）
    "CoreFacade": "facade",
    # utils
    "load_json_file": "utils",
    "load_noval_file": "utils",
    # writer
    "Segment": "writer",
    "lrc_write": "writer",
    "merge_repeated_segments": "writer",
}


def __getattr__(name):
    """惰性导出：首次访问时导入对应子模块并缓存到模块属性。"""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'core' has no attribute '{name}'")
    import importlib

    module = importlib.import_module(f".{module_name}", __name__)
    attr = getattr(module, "Facade" if name == "CoreFacade" else name)
    globals()[name] = attr  # 缓存，后续访问零开销
    return attr

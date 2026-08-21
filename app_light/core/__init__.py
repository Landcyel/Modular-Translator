"""
Translator Core — public API.

Lazy exports (PEP 562 ``__getattr__``): ``from core import X`` remains
compatible, but importing a single submodule (e.g. ``core.contracts`` /
``core.system_config``) no longer triggers a full-package aggregate import.
Aggregate imports pull in the heavy openai / numpy chains (measured ≈2.4s),
slowing the app shell's first frame; heavy modules are deferred to UI
background initialization instead.

All imports are via package-relative paths; no sys.path hacks needed.
"""

from app import torch_runtime  # noqa: F401  # ensure pluggable torch runtime selection at any core entry point (lightweight, no torch import)

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

# lazy-export mapping: name → owning submodule (relative to the core package)
_LAZY_EXPORTS = {
    # task queue (Task/TaskStatus re-exported by task_que from contracts)
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
    # facade (the actual attribute inside the submodule is named Facade)
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
    """Lazy export: import the corresponding submodule on first access and cache it on the module."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'core' has no attribute '{name}'")
    import importlib

    module = importlib.import_module(f".{module_name}", __name__)
    attr = getattr(module, "Facade" if name == "CoreFacade" else name)
    globals()[name] = attr  # cache; subsequent access is zero-cost
    return attr

"""MOSS-Transcribe-Diarize 后端封装（core 正式集成）。

两种形态：
- ``MossService``（主，进程内）：直调 ModelRunner，仿 FasterWhisperService；
- ``MossSubprocessService``（备用）：web_cli 子进程常驻 + HTTP 客户端。

共同契约：``execute() -> {"segments": [...], "info": {...}}``，单说话人场景
由 ``single_speaker`` 开关（prompt 抑制 + force_single_speaker 兜底）保证
纯 S01 输出。
"""
from .lrc_export import export_lrc, format_lrc_time
from .moss_service import MossService, resolve_model_path
from .moss_subprocess import MossHttpTranscriber, MossSubprocessService
from .moss_transcriber import MossTranscriber
from .speaker_utils import force_single_speaker

__all__ = [
    "MossService",
    "MossTranscriber",
    "MossSubprocessService",
    "MossHttpTranscriber",
    "resolve_model_path",
    "force_single_speaker",
    "export_lrc",
    "format_lrc_time",
]

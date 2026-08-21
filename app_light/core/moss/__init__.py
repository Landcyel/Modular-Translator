"""MOSS-Transcribe-Diarize backend wrapper (official core integration).

Two forms:
- ``MossService`` (primary, in-process): calls ModelRunner directly, modeled on FasterWhisperService;
- ``MossSubprocessService`` (backup): resident web_cli subprocess + HTTP client.

Common contract: ``execute() -> {"segments": [...], "info": {...}}``; single-speaker scenarios
are guaranteed to produce pure S01 output by the ``single_speaker`` switch
(prompt suppression + force_single_speaker fallback).
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

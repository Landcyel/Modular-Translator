"""GsvEngine — GPT-SoVITS inference engine (replicates the inference_webui_fast.py GUI inference flow).

Replication reference (source project inference_webui_fast.py):
- Model loading: TTS_Config({"custom": {...}}) → TTS(...)  (same construction as :125-144)
- Synthesis: the inputs dict matches the GUI inference() key for key (:175-196),
              `for item in tts_pipeline.run(inputs): yield item` (:197-199)
- Stop: tts_pipeline.stop() (:463, same stop_flag semantics)
- Hot-swap: init_vits_weights / init_t2s_weights (same as change_sovits_weights / change_gpt_weights)
- Reference audio: set_ref_audio internal 3~10s hard validation + prompt_cache (TTS.py:750-832)

Runtime contract (vendored-code dependencies, satisfied automatically inside the engine):
- CWD = vendor root during import and every call (sys.path.append in TTS.py:12, the
  module-level G2PWPinyin in chinese2.py "GPT_SoVITS/text/G2PWModel", and the ckpt path
  in sv.py are all CWD-relative)
- sys.path prepends vendor root and vendor/GPT_SoVITS (bare-name imports: AR/module/text/sv/tools)
- os.environ["bert_path"] points to the roberta in models/ (chinese2.py:35 model_source)
- sv.sv_path is patched to an absolute path before constructing TTS (hard-coded relative path in sv.py:6)

Thread safety: a single instance is serialized by RLock (TTS.run mutates infer_panel attributes, so
it is not concurrency-safe); every public call saves/restores CWD, never leaking the process working directory.
"""

from __future__ import annotations

import os
import random
import sys
import threading
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from . import paths as gsv_paths
from .vendor_links import VENDOR_ROOT, ensure_vendor, ensure_links, PACKAGE_DIR


def _numpy_compat_shim() -> None:
    """numpy 2.x compatibility shim: restore removed aliases that GSV code may reference (without touching vendor source)."""
    np = __import__("numpy")
    aliases = {
        "int": int, "float": float, "bool": bool, "str": str, "complex": complex,
        "object": object, "float_": np.float64, "int_": np.int64,
        "NaN": float("nan"), "Inf": float("inf"),
    }
    with warnings.catch_warnings():
        # numpy 2.2 emits FutureWarning on access/assignment to np.str/np.object
        warnings.simplefilter("ignore", FutureWarning)
        for name, val in aliases.items():
            if not hasattr(np, name):
                setattr(np, name, val)


class GsvEngine:
    """GPT-SoVITS inference engine. Loading all models (S1+S2+BERT+CNHuBERT+SV) happens at construction."""

    def __init__(self, config: Optional[dict] = None):
        self._lock = threading.RLock()
        self._raw_config = dict(config or {})   # Raw role config values (before resolve_config; used to decide whether weights were explicitly specified)
        self._cfg = gsv_paths.resolve_config(config)
        self._vendor_root: Path = VENDOR_ROOT
        self._tts = None
        self.last_seed: Optional[int] = None
        self._released = False

        _numpy_compat_shim()
        ensure_vendor(auto_copy=True)
        # Re-confirm junctions against the resolved config (when env overrides the target)
        ensure_links({
            PACKAGE_DIR / "text/G2PWModel": Path(self._cfg["g2pw_dir"]),
            PACKAGE_DIR / "pretrained_models/sv": Path(self._cfg["sv_dir"]),
            PACKAGE_DIR / "pretrained_models/fast_langdetect": Path(self._cfg["langdetect_dir"]),
            PACKAGE_DIR / "pretrained_models/gsv-v4-pretrained": Path(self._cfg["vocoder_dir"]),
        })

        with self._vendor_ctx():
            import time as _time
            _rt_t0 = _time.perf_counter()
            self._setup_runtime()
            _setup_sec = _time.perf_counter() - _rt_t0
            self._load_models(setup_sec=_setup_sec)

    # ── Environment ──

    @contextmanager
    def _vendor_ctx(self) -> Iterator[None]:
        """Lock + CWD=vendor root, restoring CWD on exit. All vendored calls must run inside it."""
        with self._lock:
            prev = os.getcwd()
            os.chdir(self._vendor_root)
            try:
                yield
            finally:
                os.chdir(prev)

    def _setup_runtime(self) -> None:
        """One-time environment setup: sys.path + env (must run with CWD=vendor)."""
        for p in (str(self._vendor_root), str(self._vendor_root / "GPT_SoVITS")):
            if p not in sys.path:
                sys.path.insert(0, p)
        # Ensure bare-name ffmpeg calls from vendored TTS.py / tools hit the project's bundled copy
        from app.ffmpeg import ensure_ffmpeg_on_path

        ensure_ffmpeg_on_path()
        # The G2PWPinyin(model_source=...) in chinese2.py:35 reads this env
        os.environ["bert_path"] = self._cfg["bert_base_path"]
        self._patch_torchaudio_load()

    def _patch_torchaudio_load(self) -> None:
        """torchaudio.load → soundfile equivalent implementation (torchcodec is ABI-incompatible with torch 2.13).

        torchaudio 2.9+ removed set_audio_backend; load defaults to torchcodec, whose
        DLL in torchcodec 0.16 fails to load under torch 2.13. The vendored code only
        reads the reference audio via ``torchaudio.load(path)`` in TTS.py:772 (wav
        convention), so this is replaced with a soundfile implementation (dtype
        float32, same (ch, N) tensor semantics).
        """
        import torchaudio  # noqa: F401

        import soundfile as sf
        import torch

        def _load_sf(ref_audio_path):
            data, sr = sf.read(ref_audio_path, dtype="float32", always_2d=True)
            return torch.from_numpy(data.T.copy()), sr

        torchaudio.load = _load_sf

    def _import_tts(self):
        """Lazily import the vendored TTS class (executed on first construction)."""
        from TTS_infer_pack.TTS import TTS, TTS_Config  # noqa: E402

        return TTS, TTS_Config

    def _load_models(self, setup_sec: float | None = None) -> None:
        import time
        from app.log import log

        t0 = time.perf_counter()
        TTS, TTS_Config = self._import_tts()
        t1 = time.perf_counter()

        import sv  # noqa: E402 — vendored sv.py (module-level already loaded with the TTS import)

        # sv.py:6 hard-coded relative path → absolute path (SV.__init__ reads the module global at runtime)
        sv.sv_path = self._cfg["sv_path"]

        custom = {
            "device": self._cfg["device"],
            "is_half": self._cfg["is_half"],
            "version": self._cfg["version"],
            "t2s_weights_path": self._cfg["t2s_weights_path"],
            "vits_weights_path": self._cfg["vits_weights_path"],
            "bert_base_path": self._cfg["bert_base_path"],
            "cnhuhbert_base_path": self._cfg["cnhuhbert_base_path"],
        }
        self._tts = TTS(TTS_Config({"custom": custom}))
        t2 = time.perf_counter()
        self._normalize_cpu_dtype()
        log.record(
            "info",
            f"[gsv] 加载计时: torch/CUDA初始化+setup={setup_sec or 0:.1f}s, "
            f"import_tts(库导入+jit+G2PW)={t1 - t0:.1f}s, "
            f"TTS构造(权重+模型构建)={t2 - t1:.1f}s, "
            f"引擎总(含setup)={t2 - t0 + (setup_sec or 0):.1f}s",
        )

    def _normalize_cpu_dtype(self) -> None:
        """Normalize sub-models to float32 on CPU (weights may be fp16 slim versions).

        Official weights are fp32, but community/slim versions (e.g. the 0.19GB
        s2Gv2ProPlus.pth) are often fp16; the vendored TTS only converts per is_half
        and keeps the weight's original dtype on CPU, causing conv1d
        FloatTensor/HalfTensor mismatches. A global .float() normalization is applied
        after loading (idempotent, no effect on the CUDA path — CUDA handles each per is_half).
        """
        if str(self._cfg["device"]) == "cpu":
            for attr in ("cnhuhbert_model", "bert_model", "vits_model",
                         "t2s_model", "vocoder", "sv_model"):
                m = getattr(self._tts, attr, None)
                if m is None:
                    continue
                try:
                    m.float()
                except Exception:
                    pass  # ignore individual attributes that are not nn.Module

    # ── Inference ──

    @staticmethod
    def _abs_path(p) -> str:
        """Resolve relative paths to absolute against the caller's CWD (call before entering the vendor context). Returns empty string for None/empty."""
        if not p:
            return ""
        p = os.fspath(p)
        return os.path.abspath(p) if not os.path.isabs(p) else p

    def _make_inputs(
        self,
        text: str,
        text_lang: str,
        ref_audio_path: Optional[str],
        prompt_text: str,
        prompt_lang: str,
        return_fragment: bool,
        params: dict,
    ) -> dict:
        """Build inputs matching the GUI inference() key for key (inference_webui_fast.py:175-196)."""
        seed = params.pop("seed", -1)
        actual_seed = seed if seed not in [-1, "", None] else random.randint(0, 2**32 - 1)
        self.last_seed = actual_seed

        inputs = {
            "text": text,
            "text_lang": text_lang,
            "ref_audio_path": self._abs_path(ref_audio_path),
            "aux_ref_audio_paths": [self._abs_path(p) for p in params.pop("aux_ref_audio_paths", [])],
            "prompt_text": prompt_text,
            "prompt_lang": prompt_lang,
            "top_k": params.pop("top_k", 15),
            "top_p": params.pop("top_p", 1.0),
            "temperature": params.pop("temperature", 1.0),
            "text_split_method": params.pop("text_split_method", "cut1"),
            "batch_size": int(params.pop("batch_size", 1)),
            "speed_factor": float(params.pop("speed_factor", 1.0)),
            "split_bucket": params.pop("split_bucket", True),
            "return_fragment": return_fragment,
            "fragment_interval": params.pop("fragment_interval", 0.3),
            "seed": actual_seed,
            "parallel_infer": params.pop("parallel_infer", True),
            "repetition_penalty": params.pop("repetition_penalty", 1.35),
            "sample_steps": int(params.pop("sample_steps", 32)),
            "super_sampling": params.pop("super_sampling", False),
        }
        if params:
            raise TypeError(f"未知推理参数: {sorted(params)}")
        return inputs

    def synth_stream(
        self,
        text: str,
        text_lang: str,
        ref_audio_path: Optional[str] = None,
        prompt_text: str = "",
        prompt_lang: str = "",
        **params,
    ) -> Iterator[tuple[int, np.ndarray]]:
        """Synthesize fragment by fragment (GUI streaming semantics): each yield is (sr, np.int16).

        Supports calling stop() mid-stream to end early. Raises OSError if the
        reference audio is outside the 3~10s range.
        """
        if self._released:
            raise RuntimeError("engine already released")
        inputs = self._make_inputs(
            text, text_lang, ref_audio_path, prompt_text, prompt_lang,
            return_fragment=True, params=params,
        )
        with self._vendor_ctx():
            for item in self._tts.run(inputs):  # GUI: for item in tts_pipeline.run(inputs)
                yield item

    def synth(
        self,
        text: str,
        text_lang: str,
        ref_audio_path: Optional[str] = None,
        prompt_text: str = "",
        prompt_lang: str = "",
        **params,
    ) -> tuple[int, np.ndarray]:
        """Synthesize a whole audio clip (GUI synthesize-button semantics): returns (sr, np.int16)."""
        chunks: list[np.ndarray] = []
        sr = None
        for sr, frag in self.synth_stream(
            text, text_lang, ref_audio_path, prompt_text, prompt_lang, **params
        ):
            chunks.append(frag)
        if sr is None or not chunks:
            raise RuntimeError("合成未产出任何音频片段")
        return sr, np.concatenate(chunks)

    def synth_to_file(
        self,
        text: str,
        out_path: str | Path,
        text_lang: str = "zh",
        ref_audio_path: Optional[str] = None,
        prompt_text: str = "",
        prompt_lang: str = "ja",
        **params,
    ) -> dict:
        """Synthesize and write a wav (int16): returns {audio_path, sample_rate, duration}."""
        import soundfile

        sr, audio = self.synth(
            text, text_lang, ref_audio_path, prompt_text, prompt_lang, **params
        )
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        soundfile.write(str(out_path), audio, sr, subtype="PCM_16")
        return {
            "audio_path": str(out_path),
            "sample_rate": sr,
            "duration": len(audio) / sr,
        }

    def synth_cross_speaker(
        self,
        text: str,
        text_lang: str,
        emotion_ref_audio: str,
        emotion_text: str,
        emotion_lang: str,
        role_ref_audio: str,
        **params,
    ) -> Iterator[tuple[int, np.ndarray]]:
        """Strict dual-reference: emotion audio → S1 semantics, role audio → S2 spectrum/SV (timbre anchoring).

        prompt_cache orchestration (zero vendor changes, per the cache-comparison logic in TTS.py:1131-1137):
          1. set_ref_audio(emotion audio)  → prompt_semantic/refer_spec both come from the emotion audio
          2. Save emotion_semantic
          3. set_ref_audio(role audio)     → refer_spec + 16k audio (SV recomputed each run) anchors the role
          4. Write back cache["prompt_semantic"] → S1 keeps using the emotion semantics
          5. run(ref=role path, prompt_text=emotion text) → path hits cache without recompute, S1 emotion / S2 role

        Both references are subject to the 3~10s hard validation (TTS.py:814-816). Emotion follows the
        source audio while timbre is anchored to the role — an untested architectural proposition for
        v2ProPlus; residual timbre drift is judged empirically by SV similarity.
        """
        if self._released:
            raise RuntimeError("engine already released")
        emotion_ref = self._abs_path(emotion_ref_audio)
        role_ref = self._abs_path(role_ref_audio)
        with self._vendor_ctx():
            tts = self._tts
            tts.set_ref_audio(emotion_ref)
            emotion_semantic = tts.prompt_cache["prompt_semantic"]
            tts.set_ref_audio(role_ref)
            tts.prompt_cache["prompt_semantic"] = emotion_semantic
            inputs = self._make_inputs(
                text, text_lang, role_ref, emotion_text, emotion_lang,
                return_fragment=True, params=params,
            )
            for item in tts.run(inputs):  # GUI: for item in tts_pipeline.run(inputs)
                yield item

    # ── GUI-matching Controls ──

    def set_ref_audio(self, ref_audio_path: str) -> None:
        """Preset the reference audio (warm up prompt_cache, matching the GUI reference-audio switch semantics).

        The reference audio must be 3~10s long, otherwise OSError is raised.
        """
        ref_audio_path = self._abs_path(ref_audio_path)
        with self._vendor_ctx():
            self._tts.set_ref_audio(ref_audio_path)

    def change_sovits_weights(self, weights_path: str) -> None:
        """Hot-swap the S2 weights (same as the GUI change_sovits_weights; Pro auto-reloads SV)."""
        weights_path = self._abs_path(weights_path)
        with self._vendor_ctx():
            self._tts.init_vits_weights(weights_path)

    def change_gpt_weights(self, weights_path: str) -> None:
        """Hot-swap the S1 weights (same as the GUI change_gpt_weights)."""
        weights_path = self._abs_path(weights_path)
        with self._vendor_ctx():
            self._tts.init_t2s_weights(weights_path)

    def apply_role(self, role_cfg: Optional[dict]) -> None:
        """Hot-swap roles: only rebuild the S1/S2 weights; base models (BERT/CNHuBERT/SV/vocoder) stay resident.

        Only the two role-specific weight files S1(t2s)/S2(vits) exist; switching roles
        reuses the live engine via init_t2s_weights → init_vits_weights, dropping the
        cost from a full rebuild (10~20s) to seconds. Two pitfalls to handle:
        - CPU dtype: _normalize_cpu_dtype at construction normalizes only once; if the
          hot-swapped new S1/S2 are fp16 slim weights, apply .float() (conv1d dtype mismatch).
        - Stale reference cache: run() cache hits compare only ref_audio_path (TTS.py:1138-1144);
          after switching S2 the same-path reference audio is not recomputed
          (refer_spec/prompt_semantic were computed with the old vits) — when the role
          config carries role_ref_audio, set_ref_audio recomputes and warms it; otherwise
          set prompt_cache["ref_audio_path"]=None to force recompute on the next run.

        Assumes switching within the same version family (all current role configs are v2ProPlus);
        cross-version still requires a full rebuild.
        """
        if self._released or self._tts is None:
            raise RuntimeError("engine not ready for role switch")
        role_cfg = dict(role_cfg or {})
        t2s = role_cfg.get("t2s_weights_path") or self._cfg["t2s_weights_path"]
        vits = role_cfg.get("vits_weights_path") or self._cfg["vits_weights_path"]
        ref = role_cfg.get("role_ref_audio")
        # Paths must be absolutized outside the vendor context (per caller CWD, same pattern as synth_stream)
        t2s_abs = self._abs_path(t2s)
        vits_abs = self._abs_path(vits)
        ref_abs = self._abs_path(ref) if ref else None
        with self._vendor_ctx():
            self._tts.init_t2s_weights(t2s_abs)
            self._tts.init_vits_weights(vits_abs)
            # CPU normalization: add float32 to modules replaced by hot-swap (one-time construction normalization does not cover them)
            if str(self._cfg["device"]) == "cpu":
                for attr in ("t2s_model", "vits_model"):
                    m = getattr(self._tts, attr, None)
                    if m is None:
                        continue
                    try:
                        m.float()
                    except Exception:
                        pass
            if ref_abs:
                try:
                    self._tts.set_ref_audio(ref_abs)
                except Exception as ex:
                    import logging

                    logging.getLogger("gsv").warning(
                        "角色参考音频预热失败（run 时重算兜底）: %s", ex
                    )
            else:
                # No reference audio: invalidate the stale cache (run hit checks only ref_audio_path)
                self._tts.prompt_cache["ref_audio_path"] = None
        # Record the new role (preserving the *specified/*configured semantics of weights_status)
        self._raw_config = dict(role_cfg)
        self._cfg["t2s_weights_path"] = t2s_abs
        self._cfg["vits_weights_path"] = vits_abs

    def stop(self) -> None:
        """Request to stop the current synthesis (same as the GUI stop button: sets stop_flag, run ends early).

        Does not hold the lock — must be called while another thread (the one consuming
        synth_stream) is executing; stop_flag is a plain bool and atomicity is
        guaranteed by the GIL.
        """
        tts = self._tts
        if tts is not None:
            tts.stop()

    def release(self) -> None:
        """Release models and GPU memory (finally semantics: del + empty_cache)."""
        if self._released:
            return
        with self._vendor_ctx():
            tts, self._tts = self._tts, None
            if tts is not None:
                try:
                    tts.empty_cache()
                except Exception:
                    pass
        self._released = True

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass

    # ── Properties ──

    @property
    def version(self) -> str:
        return self._cfg["version"]

    @property
    def device(self) -> str:
        return self._cfg["device"]

    @property
    def config(self) -> dict:
        return dict(self._cfg)

    def weights_status(self) -> dict:
        """Return the S1/S2 weight loading status, for service startup logs to judge whether role weights are loaded.

        - ``*_specified``: whether the role config explicitly specified this weight path.
        - ``*_configured``: the absolute weight path resolved from the role config (default path when unspecified).
        - ``*_used``: the weight path TTS_Config actually used (including the result after default fallback).
        - ``*_role_loaded``: whether the weight actually used is the one specified by the role config.
        """
        cfg = self._cfg
        raw = self._raw_config
        tts = self._tts
        t2s_used = getattr(getattr(tts, "configs", None), "t2s_weights_path", None)
        vits_used = getattr(getattr(tts, "configs", None), "vits_weights_path", None)

        def _same(a, b) -> bool:
            if not a or not b:
                return False
            return os.path.normcase(os.path.normpath(str(a))) == os.path.normcase(os.path.normpath(str(b)))

        return {
            "t2s_specified": "t2s_weights_path" in raw and bool(raw.get("t2s_weights_path")),
            "vits_specified": "vits_weights_path" in raw and bool(raw.get("vits_weights_path")),
            "t2s_configured": cfg.get("t2s_weights_path"),
            "vits_configured": cfg.get("vits_weights_path"),
            "t2s_used": t2s_used,
            "vits_used": vits_used,
            "t2s_role_loaded": _same(t2s_used, cfg.get("t2s_weights_path")),
            "vits_role_loaded": _same(vits_used, cfg.get("vits_weights_path")),
        }

    def __repr__(self):
        return f"<GsvEngine version={self.version} device={self.device}>"

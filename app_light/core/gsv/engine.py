"""GsvEngine —— GPT-SoVITS 推理引擎（复刻 inference_webui_fast.py GUI 推理流程）。

复刻对照（源项目 inference_webui_fast.py）:
- 模型加载 : TTS_Config({"custom": {...}}) → TTS(...)  （:125-144 同款构造）
- 合成     : inputs 字典与 GUI inference() 逐键一致（:175-196），
             `for item in tts_pipeline.run(inputs): yield item`（:197-199）
- 停止     : tts_pipeline.stop()（:463 同款 stop_flag 语义）
- 热切换   : init_vits_weights / init_t2s_weights（change_sovits_weights / change_gpt_weights 同款）
- 参考音频 : set_ref_audio 内部 3~10s 硬校验 + prompt_cache（TTS.py:750-832）

运行契约（vendored 代码依赖，引擎内自动满足）:
- import 与每次调用期间 CWD = vendor 根（TTS.py:12 的 sys.path.append、chinese2.py
  模块级 G2PWPinyin 的 "GPT_SoVITS/text/G2PWModel"、sv.py 的 ckpt 路径均 CWD 相对）
- sys.path 前插 vendor 根与 vendor/GPT_SoVITS（裸名 import: AR/module/text/sv/tools）
- os.environ["bert_path"] 指向 models/ 的 roberta（chinese2.py:35 model_source）
- sv.sv_path 在构造 TTS 前打补丁为绝对路径（sv.py:6 硬编码的相对路径）

线程安全: 单实例由 RLock 串行化（TTS.run 改写 infer_panel 属性，不可并发）；
每次公开调用 save/restore CWD，不泄漏进程工作目录。
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
    """numpy 2.x 兼容垫片: 补回 GSV 代码可能引用的已删别名（不动 vendor 源码）。"""
    np = __import__("numpy")
    aliases = {
        "int": int, "float": float, "bool": bool, "str": str, "complex": complex,
        "object": object, "float_": np.float64, "int_": np.int64,
        "NaN": float("nan"), "Inf": float("inf"),
    }
    with warnings.catch_warnings():
        # numpy 2.2 对 np.str/np.object 的访问/赋值会发 FutureWarning
        warnings.simplefilter("ignore", FutureWarning)
        for name, val in aliases.items():
            if not hasattr(np, name):
                setattr(np, name, val)


class GsvEngine:
    """GPT-SoVITS 推理引擎。构造即加载全部模型（S1+S2+BERT+CNHuBERT+SV）。"""

    def __init__(self, config: Optional[dict] = None):
        self._lock = threading.RLock()
        self._raw_config = dict(config or {})   # 角色配置原始值（resolve_config 前，用于判断是否显式指定权重）
        self._cfg = gsv_paths.resolve_config(config)
        self._vendor_root: Path = VENDOR_ROOT
        self._tts = None
        self.last_seed: Optional[int] = None
        self._released = False

        _numpy_compat_shim()
        ensure_vendor(auto_copy=True)
        # junction 按解析后的配置再确认一次（env 覆盖了目标时）
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

    # ---------------------------------------------------------------- 环境

    @contextmanager
    def _vendor_ctx(self) -> Iterator[None]:
        """锁 + CWD=vendor 根，退出时恢复 CWD。所有 vendored 调用必须在其内。"""
        with self._lock:
            prev = os.getcwd()
            os.chdir(self._vendor_root)
            try:
                yield
            finally:
                os.chdir(prev)

    def _setup_runtime(self) -> None:
        """一次性环境准备: sys.path + env（须在 CWD=vendor 下执行）。"""
        for p in (str(self._vendor_root), str(self._vendor_root / "GPT_SoVITS")):
            if p not in sys.path:
                sys.path.insert(0, p)
        # 确保 vendored TTS.py / tools 的裸名 ffmpeg 调用命中项目自带副本
        from app.ffmpeg import ensure_ffmpeg_on_path

        ensure_ffmpeg_on_path()
        # chinese2.py:35 的 G2PWPinyin(model_source=...) 读取该 env
        os.environ["bert_path"] = self._cfg["bert_base_path"]
        self._patch_torchaudio_load()

    def _patch_torchaudio_load(self) -> None:
        """torchaudio.load → soundfile 等价实现（torchcodec 与 torch 2.13 ABI 不兼容）。

        torchaudio 2.9+ 移除 set_audio_backend，load 默认走 torchcodec，而
        torchcodec 0.16 的 DLL 在 torch 2.13 下加载失败；vendored 仅在
        TTS.py:772 以 ``torchaudio.load(path)`` 读取参考音频（wav 约定），
        此处替换为 soundfile 实现（dtype float32, (ch, N) 张量语义一致）。
        """
        import torchaudio  # noqa: F401

        import soundfile as sf
        import torch

        def _load_sf(ref_audio_path):
            data, sr = sf.read(ref_audio_path, dtype="float32", always_2d=True)
            return torch.from_numpy(data.T.copy()), sr

        torchaudio.load = _load_sf

    def _import_tts(self):
        """惰性 import vendored TTS 类（首次构造时执行）。"""
        from TTS_infer_pack.TTS import TTS, TTS_Config  # noqa: E402

        return TTS, TTS_Config

    def _load_models(self, setup_sec: float | None = None) -> None:
        import time
        from app.log import log

        t0 = time.perf_counter()
        TTS, TTS_Config = self._import_tts()
        t1 = time.perf_counter()

        import sv  # noqa: E402 — vendored sv.py（模块级已随 TTS import 加载）

        # sv.py:6 硬编码的相对路径 → 绝对路径（SV.__init__ 运行时读取模块全局）
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
        """CPU 模式统一子模型为 float32（权重可能是 fp16 精简版）。

        官方权重为 fp32，但社区/精简版（如 0.19GB 的 s2Gv2ProPlus.pth）常为
        fp16；vendored TTS 仅按 is_half 转换、CPU 时保留权重原始 dtype，
        导致 conv1d 报 FloatTensor/HalfTensor 不匹配。此处加载后整体 .float()
        归一化（幂等，不影响 CUDA 路径——CUDA 按 is_half 各自处理）。
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
                    pass  # 个别属性非 nn.Module 时忽略

    # ---------------------------------------------------------------- 推理

    @staticmethod
    def _abs_path(p) -> str:
        """相对路径按调用方 CWD 绝对化（进入 vendor 上下文前调用）。None/空返回空串。"""
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
        """构造与 GUI inference() 逐键一致的 inputs（inference_webui_fast.py:175-196）。"""
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
        """逐分片合成（GUI 流式语义）: 每个 yield 为 (sr, np.int16)。

        支持中途调用 stop() 提前结束。参考音频超 3~10s 范围抛 OSError。
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
        """合成整段音频（GUI 合成按钮语义）: 返回 (sr, np.int16)。"""
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
        """合成并写 wav（int16）: 返回 {audio_path, sample_rate, duration}。"""
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
        """严格双参考: 情绪音频 → S1 语义, 角色音频 → S2 谱/SV（音色锚定）。

        prompt_cache 编排（零 vendor 改动，对照 TTS.py:1131-1137 缓存比对逻辑）:
          1. set_ref_audio(情绪音频)   → prompt_semantic/refer_spec 均来自情绪音频
          2. 保存 emotion_semantic
          3. set_ref_audio(角色音频)   → refer_spec + 16k 音频（SV 每次 run 重算）锚定角色
          4. 回写 cache["prompt_semantic"] → S1 继续用情绪语义
          5. run(ref=角色路径, prompt_text=情绪文本) → 路径命中缓存不重算，S1 情绪 / S2 角色

        两个参考均受 3~10s 硬校验（TTS.py:814-816）。情绪跟随源音频、音色锚定角色——
        v2ProPlus 未实测的架构命题，音色残余漂移由 SV 相似度实证判定。
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

    # ---------------------------------------------------------------- GUI 同款控制

    def set_ref_audio(self, ref_audio_path: str) -> None:
        """预置参考音频（预热 prompt_cache，对应 GUI 切换参考音频语义）。

        参考音频时长须在 3~10s 内，否则抛 OSError。
        """
        ref_audio_path = self._abs_path(ref_audio_path)
        with self._vendor_ctx():
            self._tts.set_ref_audio(ref_audio_path)

    def change_sovits_weights(self, weights_path: str) -> None:
        """热切换 S2 权重（GUI change_sovits_weights 同款，Pro 自动重载 SV）。"""
        weights_path = self._abs_path(weights_path)
        with self._vendor_ctx():
            self._tts.init_vits_weights(weights_path)

    def change_gpt_weights(self, weights_path: str) -> None:
        """热切换 S1 权重（GUI change_gpt_weights 同款）。"""
        weights_path = self._abs_path(weights_path)
        with self._vendor_ctx():
            self._tts.init_t2s_weights(weights_path)

    def apply_role(self, role_cfg: Optional[dict]) -> None:
        """热切换角色：仅重建 S1/S2 权重，基础模型（BERT/CNHuBERT/SV/vocoder）常驻。

        角色专属的只有 S1(t2s)/S2(vits) 两个权重文件；换角色复用现役引擎
        依次 init_t2s_weights → init_vits_weights，耗时从全量重建（10~20s）
        降至秒级。两个必须处理的坑：
        - CPU dtype：构造时 _normalize_cpu_dtype 只归一一次，热切换替换出的
          新 S1/S2 若为 fp16 精简权重，需补 .float()（conv1d dtype 不匹配）。
        - 参考缓存陈旧：run() 缓存命中只比 ref_audio_path（TTS.py:1138-1144），
          换 S2 后同路径参考音频不会重算（refer_spec/prompt_semantic 按旧 vits
          计算）——角色配置带 role_ref_audio 时 set_ref_audio 重算预热；否则
          置 prompt_cache["ref_audio_path"]=None 强制下次 run 重算。

        假定同 version 家族切换（当前角色配置全部 v2ProPlus）；跨版本仍需
        全量重建。
        """
        if self._released or self._tts is None:
            raise RuntimeError("engine not ready for role switch")
        role_cfg = dict(role_cfg or {})
        t2s = role_cfg.get("t2s_weights_path") or self._cfg["t2s_weights_path"]
        vits = role_cfg.get("vits_weights_path") or self._cfg["vits_weights_path"]
        ref = role_cfg.get("role_ref_audio")
        # 路径绝对化须在 vendor 上下文之外（按调用方 CWD，与 synth_stream 同模式）
        t2s_abs = self._abs_path(t2s)
        vits_abs = self._abs_path(vits)
        ref_abs = self._abs_path(ref) if ref else None
        with self._vendor_ctx():
            self._tts.init_t2s_weights(t2s_abs)
            self._tts.init_vits_weights(vits_abs)
            # CPU 归一：热切换替换出的新模块补 float32（构造时一次性归一不覆盖）
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
                # 无参考音频：使旧缓存失效（run 命中判断只比 ref_audio_path）
                self._tts.prompt_cache["ref_audio_path"] = None
        # 记录新角色（weights_status 的 *specified/*configured 语义保持）
        self._raw_config = dict(role_cfg)
        self._cfg["t2s_weights_path"] = t2s_abs
        self._cfg["vits_weights_path"] = vits_abs

    def stop(self) -> None:
        """请求停止当前合成（GUI stop 按钮同款: 置 stop_flag，run 提前结束）。

        不持锁 —— 必须在其他线程（正在消费 synth_stream 的线程）执行期间调用；
        stop_flag 为普通布尔，GIL 保证原子性。
        """
        tts = self._tts
        if tts is not None:
            tts.stop()

    def release(self) -> None:
        """释放模型与显存（finally 语义: del + empty_cache）。"""
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

    # ---------------------------------------------------------------- 属性

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
        """返回 S1/S2 权重加载状态，供服务启动日志判断“角色权重是否加载”。

        - ``*_specified``：角色配置是否显式指定了该权重路径。
        - ``*_configured``：角色配置解析后的绝对权重路径（未指定时取默认路径）。
        - ``*_used``：TTS_Config 实际采用的权重路径（含默认回退后的结果）。
        - ``*_role_loaded``：实际使用的权重是否就是角色配置指定的权重。
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

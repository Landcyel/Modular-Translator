"""MOSS 进程内转写执行器（直调 ModelRunner，token 级进度/取消 + 实时段预览）。

MOSS 库（``moss_transcribe_diarize`` + ``transformers>=5.6``）随主环境安装；
本类仿 ``core.executor.Transcriber`` 的契约，进度回调四参语义与
``TranscriptionTaskQueue`` 对齐（pos/total/speed/segs）。

``StreamingModelRunner``（SUPPORTS_PARTIAL_TEXT）可用时，pos/total 为
音频时间轴秒数、segs 为运行中已确认的字幕段——与 Whisper 的进度条与
预览逻辑完全一致；老版 runner 回退 token 比例语义（unit=ratio）。

长音频保护（CUDA OOM）：MOSS 的 Qwen3 解码器把整段音频的全部
audio-token 一次性塞进序列做全注意力（12.5 token/s），20 分钟音频
≈ 1.6 万 token，仅注意力分数张量就 ≈ 15 GiB，6 GB 显卡必然 OOM。
切块方案（三层保障，详见 PLANS/gsv-moss/plan-moss-long-audio-chunking.md）：

1. **显存预算窗口**（``vram_auto_fit``）：按空闲显存解二次峰值模型
   ``峰值 ≈ W + C1·t + C2·t²`` 反推单窗安全时长，钳制在
   ``[min_window_sec, max_audio_sec]``——显存大的卡自动放宽窗口、
   小的卡自动收敛，无需手工调参；``max_audio_sec`` 为硬上限。
2. **静音感知边界**（``silence_boundary``）：每个候选切点在
   ``[target−boundary_lookback_sec, target]`` 带内选最长静音段
   （能量包络，默认 0.35s 静音起判），切点落在自然停顿上，不截断
   正常说话；带内无静音则退回算术硬切并打 hard 标记。窗口只缩不长，
   单窗时长硬性 ≤ 显存预算值。
3. **边界文本修复 + 运行时 OOM 退避**：硬切边界处被截断的句尾段与
   下一窗口续接的句头段按文本相似度配对缝合为完整段；单窗转写仍
   OOM 时自动缩窗（×0.7，下限 45s）重规划剩余音频并重试。

各窗口段打回全局时间轴后去重合并；每窗口 ``max_new_tokens`` 同时按
窗口时长收敛，防止失控生成。
"""
from __future__ import annotations

import difflib
import shutil
import subprocess
import time
from functools import partial
from pathlib import Path
from typing import Callable, Optional

from app.paths import project_root
from ..contracts import CancelledError
from ..executor import Executor, _wait_paused
from ..writer import format_lrc_time
from .audio_utils import probe_duration
from .silence_probe import find_silence_cut, load_band_rms
from .speaker_utils import force_single_speaker

# 长音频分窗默认值：180s 窗口（约 2340 个 decoder token）在 6 GB 显卡上
# 注意力峰值 ≈ 350 MB，留足模型权重/KV/激活余量；重叠 10s 用于边界续接。
DEFAULT_WINDOW_SEC = 180.0
DEFAULT_OVERLAP_SEC = 10.0
# 每窗口生成预算：16 token/s（日/中密集语音 + 时间戳/说话人开销的宽松上限）
_WINDOW_TOKEN_RATE = 16

# ── 显存预算常量（bf16，Qwen3 28 层全注意力，约 13 token/s）────────
# 峰值模型：peak(t) ≈ W + C1·t + C2·t²（t = 窗口秒数）
#   C2 = 64.5 B/token² × (13 token/s)² ≈ 10900 B/s²
#        （注意力分数/softmax/掩码等二次瞬时分配，跨层经分配器累积）
#   C1 = 114688 B/token × 13 token/s ≈ 1.49 MB/s（KV 缓存线性项）
# 校准锚点：1217s → 峰值 ≈ 15.4 GiB（6 GB 卡 OOM）；180s → ≈ 620 MB。
_ATTN_BYTES_PER_SEC2 = 10900.0
_KV_BYTES_PER_SEC = 1_490_000.0
_WEIGHTS_ESTIMATE_BYTES = 1_365_000_000  # 懒加载未就绪时的权重估算（Qwen3-0.6B bf16 + 编码器）
_FIXED_SLACK_BYTES = 512 * 1024 * 1024   # mel 特征/输出 logits/分配器碎片等固定开销
_DEFAULT_MIN_WINDOW_SEC = 60.0
# OOM 退避：缩窗系数 / 单窗下限 / 最大退避次数
_OOM_SHRINK = 0.7
_OOM_FLOOR_SEC = 45.0
_OOM_MAX_RETREATS = 3
# 硬切边界修复容差（秒）：段尾距切点 1.2s 内视为可能被截断，
# 段头在切点后 0.5s 内视为切点处续接。
_HARD_CUT_TAIL_TOL = 1.2
_HARD_CUT_HEAD_TOL = 0.5


class MossTranscriber(Executor):
    """Execute transcription using an externally-managed MOSS ModelRunner.

    Usage via TaskQueue::
        runner = ModelRunner(model_path, device="auto", dtype="bf16")
        tx = MossTranscriber(runner, defaults=config)
        # TaskQueue calls tx.execute(task, progress_callback, cancel_event)

    参数分四类（``defaults`` 为服务级参数，任务级 ``configs["args"]`` 覆盖之）：
    - 服务参数（configs/models/moss/default.json）：model_path / device / dtype（加载时生效）
    - 转写参数：max_new_tokens / max_len / decoding / temperature / top_p /
      top_k / single_speaker（temperature/top_p/top_k 仅 decoding="sample" 时生效）
    - 长音频分窗：max_audio_sec（单窗硬上限）/ overlap_sec / vram_auto_fit /
      vram_safety_ratio / min_window_sec / silence_boundary /
      silence_min_sec / boundary_lookback_sec（见模块 docstring）
    - prompt：转写提示词（服务默认，任务 args 可覆盖）
    - hotwords：任务级 ``configs["hotwords"]``（configs/transcribe/hotwords/*.json），
      按官方配方附加到提示词末尾（"热词提示：词1, 词2…"）
    """

    def __init__(self, runner, defaults: Optional[dict] = None,
                 on_first_load: Optional[Callable[[Optional[str], Optional[str]], None]] = None):
        self.runner = runner
        self.defaults = defaults or {}
        self.on_first_load = on_first_load  # 首次转写（模型实际加载）后回调(device, dtype)
        self._device_logged = False   # 首次转写完成后补记真实设备（ModelRunner 懒加载）

    def execute(
        self,
        task,
        progress_callback: Optional[Callable[[float, float, float, Optional[list]], None]] = None,
        cancel_event: Optional[object] = None,
    ) -> dict:
        """Run transcription from *task*.

        Task 契约::
            task.file_path             音频文件路径
            task.configs["args"]       转写参数（max_new_tokens/max_len/decoding…，
                                        路径 → JSON dict，覆盖服务 defaults）

        Returns {"segments": [...], "info": TranscriptionResult.to_dict()}.

        进度回调与 Whisper ``Transcriber`` 统一为
        ``(pos, total, speed, payload)``：pos/total 为音频时间轴秒数，
        pos 为最新已确认段尾（与 Whisper 的 ``seg.end`` 同义，首段前为 0），
        完成时补发一次 pos=total 使进度满格；payload 携带
        status/generated_tokens 与运行中已确认的 segments
        （仅 ``SUPPORTS_PARTIAL_TEXT`` 的 runner 提供）。
        """
        cfg = self._resolve_task(task)
        args = cfg.get("transcribe_config") or {}
        merged = {**self.defaults, **args}
        hotwords = cfg.get("hotwords")
        pause_event = getattr(task, "_pause_event", None)
        audio_path = cfg["audio_path"]
        total_sec = probe_duration(audio_path)
        started = time.time()
        state = {
            "stage": 0.0,          # 比例进度回退值（时长探测失败时使用）
            "tokens": 0,           # 当前窗口生成 token 数
            "tokens_base": 0,      # 已完成窗口的 token 累计（分窗模式）
            "pos_base": 0.0,       # 当前窗口在全局时间轴上的起点
            "pos_floor": 0.0,      # 已发射进度的单调下界（分窗切换防回退）
            "gen_started": None,   # 解码开始时刻（首个生成 token）
        }

        def _check_cancel():
            if cancel_event is not None and cancel_event.is_set():
                raise CancelledError(task.id)
            if not _wait_paused(pause_event, cancel_event):
                raise CancelledError(task.id)

        def _emit(status: str, stage_progress: float, generated_tokens: int,
                  partial_text: Optional[str] = None):
            """统一进度发射：取消/暂停检查点 + Whisper 同构四参回调。

            token 级进度更新已删除（进度只由已确认段驱动）：无已确认段且
            transcribing 状态一律不发回调，避免把已推进的进度打回 0% 造成
            闪烁；loading_model 等加载里程碑保留 pos=当前窗口起点（加载期
            进度本为 0）。
            """
            _check_cancel()
            state["stage"] = float(stage_progress or 0.0)
            if generated_tokens is not None:
                state["tokens"] = int(generated_tokens)
            tokens_total = int(state.get("tokens_base") or 0) + int(state["tokens"] or 0)
            if state["tokens"] > 0 and state["gen_started"] is None:
                # 与 Whisper 的 start_wall 对齐：倍速自解码开始（首个生成
                # token）起计，排除模型懒加载耗时。
                state["gen_started"] = time.time()
            if progress_callback is None:
                return

            now = time.time()
            segments = None
            if partial_text:
                segments = self._segments_from_text(
                    partial_text, merged,
                    offset=float(state.get("pos_base") or 0.0),
                    apply_single=False,
                )
                if segments:
                    state["pos_floor"] = max(
                        state.get("pos_floor") or 0.0,
                        max(float(s["end"]) for s in segments),
                    )
            payload = {"status": status, "generated_tokens": tokens_total}
            if segments:
                payload["segments"] = segments
            if total_sec:
                if segments:
                    # 与 Whisper 同构：pos = 最新已确认段尾（真实时间轴），
                    # 倍速自解码开始起计，与 Whisper pos/elapsed 口径一致。
                    pos = max(float(s["end"]) for s in segments)
                    speed = None
                    if state["gen_started"] is not None:
                        speed = pos / max(now - state["gen_started"], 1e-6)
                    progress_callback(pos, total_sec, speed, payload)
                elif status != "transcribing":
                    # 加载期里程碑（loading_model 等）：进度保持在当前窗口
                    # 起点之上，仅刷新 status 文本（加载期无正常进度，不闪烁）。
                    pos = max(
                        float(state.get("pos_base") or 0.0),
                        float(state.get("pos_floor") or 0.0),
                    )
                    progress_callback(pos, total_sec, None, payload)
                # 无已确认段且 transcribing：不发（token 级进度已删除；加载
                # 里程碑与偶发段解析失败均不再把进度打回 0%）
            else:
                # 时长探测失败：回退 token 比例语义（unit=ratio 供 UI 区分）
                payload["unit"] = "ratio"
                progress_callback(state["stage"], 1.0, None, payload)

        def _emit_window_end(window_end: float):
            """分窗模式下每个窗口完成后补发一次进度（单调推进到窗口尾）。"""
            _check_cancel()
            if progress_callback is None:
                return
            pos = max(
                min(float(window_end), float(total_sec or window_end)),
                float(state.get("pos_floor") or 0.0),
            )
            pos = min(pos, float(total_sec or pos))
            state["pos_floor"] = pos
            payload = {
                "status": "transcribing",
                "generated_tokens": int(state.get("tokens_base") or 0) + int(state["tokens"] or 0),
            }
            if total_sec:
                speed = None
                if state["gen_started"] is not None:
                    speed = pos / max(time.time() - state["gen_started"], 1e-6)
                progress_callback(pos, total_sec, speed, payload)
            else:
                payload["unit"] = "ratio"
                progress_callback(1.0, 1.0, None, payload)

        def _emit_final(seg_dicts: list, generated_total: int):
            """收尾进度：与 Whisper 末段 end≈duration 对齐，进度补满 100%。

            队列在 execute 返回后才翻转 status，故完成瞬间进度条可见满格。
            """
            _check_cancel()
            if progress_callback is None:
                return
            payload = {
                "status": "transcribing",
                "generated_tokens": int(generated_total or 0),
            }
            if seg_dicts:
                payload["segments"] = seg_dicts
            if total_sec:
                speed = None
                if state["gen_started"] is not None:
                    speed = total_sec / max(time.time() - state["gen_started"], 1e-6)
                progress_callback(total_sec, total_sec, speed, payload)
            else:
                payload["unit"] = "ratio"
                progress_callback(1.0, 1.0, None, payload)

        def _status(status: str, progress: float, generated_tokens: int):
            _emit(status, progress, generated_tokens)

        def _partial_text(partial_text: str, generated_tokens: int):
            _emit("transcribing", state["stage"], generated_tokens, partial_text)

        transcribe_kwargs = self._build_transcribe_kwargs(merged, hotwords)
        transcribe_kwargs["status_callback"] = _status
        if getattr(self.runner, "SUPPORTS_PARTIAL_TEXT", False):
            transcribe_kwargs["partial_text_callback"] = _partial_text
        window_sec = self._resolve_window_sec(merged)
        envelope_getter = None
        if (window_sec and total_sec and total_sec > window_sec + 1e-3
                and merged.get("silence_boundary", True)):
            envelope_getter = partial(load_band_rms, audio_path)
        windows, hard_flags = self._plan_windows_with_flags(
            total_sec, merged, envelope_getter, window_sec,
        )

        if windows:
            overlap = self._as_float(merged.get("overlap_sec"), DEFAULT_OVERLAP_SEC)
            hard_count = sum(1 for f in hard_flags if f)
            self._log(
                "info",
                f"[transcribe] 长音频分窗转写：总长={total_sec:.1f}s，"
                f"窗口={len(windows)}，单窗≤{max(e - s for s, e in windows):.1f}s，"
                f"重叠={overlap:.1f}s，静音切分={len(windows) - hard_count}，"
                f"硬切={hard_count}（硬切边界自动文本修复）",
            )
            seg_dicts, info = self._transcribe_windowed(
                task=task,
                audio_path=audio_path,
                windows=windows,
                hard_flags=hard_flags,
                total_sec=total_sec,
                merged=merged,
                base_kwargs=transcribe_kwargs,
                state=state,
                started=started,
                check_cancel=_check_cancel,
                emit_window_end=_emit_window_end,
                envelope_getter=envelope_getter,
                window_sec=window_sec,
            )
            generated_total = int((info or {}).get("generated_tokens") or 0)
        else:
            try:
                result = self.runner.transcribe(audio_path, **transcribe_kwargs)
            except CancelledError:
                self._empty_cuda_cache()  # 取消后释放 allocator 缓存
                raise
            self._note_first_load(started)
            # 最终结果与实时预览共用同一段切分管线，保证预览末帧 = 最终结果。
            seg_dicts = self._segments_from_text(result.text, merged)
            info = result.to_dict() if hasattr(result, "to_dict") else dict(result or {})
            if not isinstance(info, dict):
                info = {}
            generated_total = int(
                info.get("generated_tokens", state["tokens"]) or 0
            )

        _emit_final(seg_dicts, generated_total)
        return {"segments": seg_dicts, "info": info}

    # ── 长音频分窗 ─────────────────────────────────────────────

    def _resolve_window_sec(self, merged: dict) -> Optional[float]:
        """显存预算 → 单窗安全时长（秒）；None/≤0 = 禁用分窗。

        ``vram_auto_fit=false``、非 CUDA 设备或探测失败时回落
        ``max_audio_sec``（既有固定窗口行为）。CUDA 下按二次峰值模型
        ``C2·t² + C1·t = 预算`` 反推时长（预算 = 空闲显存 ×
        ``vram_safety_ratio`` − 固定开销 − 权重估算，模型已加载时
        权重已在空闲显存中扣除、不再重复扣减），结果钳制在
        ``[min_window_sec, max_audio_sec]``。
        """
        max_sec = self._as_float(merged.get("max_audio_sec"), DEFAULT_WINDOW_SEC)
        if max_sec <= 0:
            return None
        if not merged.get("vram_auto_fit", True):
            return max_sec
        free = self._probe_free_vram(merged)
        if free is None:
            return max_sec
        ratio = self._as_float(merged.get("vram_safety_ratio"), 0.7)
        ratio = max(0.1, min(ratio, 0.95))
        budget = free * ratio - _FIXED_SLACK_BYTES
        # 最低预算保障（GB，0=不设下限）：显存较小时也按指定预算选窗，
        # 优先把窗口开大（宁可触发 OOM 缩窗兜底，也不让窗口过小）
        min_gb = self._as_float(merged.get("min_vram_budget_gb"), 0.0)
        if min_gb > 0:
            budget = max(budget, int(min_gb * 1024 ** 3))
        if getattr(self.runner, "_model", None) is None:
            budget -= _WEIGHTS_ESTIMATE_BYTES  # 懒加载：空闲显存尚未扣除权重
        min_sec = min(
            self._as_float(merged.get("min_window_sec"), _DEFAULT_MIN_WINDOW_SEC),
            max_sec,
        )
        if budget <= 0:
            return max(min_sec, 1.0)  # 预算极低：保底最小窗，交给 OOM 退避兜底
        import math

        disc = _KV_BYTES_PER_SEC ** 2 + 4.0 * _ATTN_BYTES_PER_SEC2 * budget
        sec = (-_KV_BYTES_PER_SEC + math.sqrt(disc)) / (2.0 * _ATTN_BYTES_PER_SEC2)
        return max(min(sec, max_sec), min_sec)

    def _probe_free_vram(self, merged: dict) -> Optional[int]:
        """CUDA 空闲显存探测（字节）；非 CUDA 配置 / 探测失败返回 None。"""
        device_cfg = str(merged.get("device") or "auto").lower()
        if device_cfg == "cpu":
            return None
        actual = getattr(self.runner, "_device", None)
        if actual is not None and actual.type != "cuda":
            return None
        try:
            import torch
        except Exception:
            return None
        try:
            if not torch.cuda.is_available():
                return None
            device_idx = None
            if actual is not None and actual.type == "cuda":
                device_idx = getattr(actual, "index", None)
            elif device_cfg.startswith("cuda:"):
                device_idx = int(device_cfg.split(":", 1)[1])
            free, _total = (
                torch.cuda.mem_get_info(device_idx)
                if device_idx is not None
                else torch.cuda.mem_get_info()
            )
            return int(free)
        except Exception:
            return None

    def _plan_windows(self, total_sec: Optional[float], merged: dict) -> list:
        """兼容入口：超过 max_audio_sec 的音频 → 算术滑动窗口（无静音探测）。

        相邻窗口重叠 overlap_sec；总长未超阈值 / 时长未知 / 阈值为 0 时
        返回空列表（调用方走单次整段转写路径）。
        """
        return self._plan_windows_with_flags(total_sec, merged)[0]

    def _plan_windows_with_flags(self, total_sec: Optional[float], merged: dict,
                                 envelope_getter=None, window_sec: Optional[float] = None,
                                 start_from: float = 0.0) -> tuple[list, list]:
        """静音感知滑动窗口 → ([(start, end), ...], [hard_flag, ...])。

        单窗长度 = ``window_sec``（显存预算值）或 ``max_audio_sec``。
        每个内部边界在 ``[target − lookback, target]`` 带内（target =
        窗首 + 窗长）寻找最佳静音段作为切点：切点只可能提前、不会推后，
        **单窗时长硬性 ≤ 预算值**（显存安全的构造性保证）。带内无静音
        → 算术硬切（flag=True，交由边界文本修复兜底）。

        ``start_from`` 用于 OOM 退避时从失败窗口起重规划剩余音频；
        hard_flag 与窗口对齐：flag[i] = 窗口 i 右边界是否为硬切
        （末窗口恒为 False）。
        """
        max_sec = (
            window_sec
            if window_sec is not None
            else self._as_float(merged.get("max_audio_sec"), DEFAULT_WINDOW_SEC)
        )
        overlap = self._as_float(merged.get("overlap_sec"), DEFAULT_OVERLAP_SEC)
        if total_sec is None or max_sec <= 0:
            return [], []
        start_from = max(0.0, float(start_from))
        if start_from <= 0 and total_sec <= max_sec + 1e-3:
            return [], []
        overlap = max(0.0, min(overlap, max_sec * 0.5))
        if max_sec - overlap < 5.0:
            return [], []
        lookback = max(0.0, self._as_float(merged.get("boundary_lookback_sec"), 30.0))
        lookback = min(lookback, max_sec * 0.5)
        silence_min = max(0.15, self._as_float(merged.get("silence_min_sec"), 0.35))
        use_silence = bool(merged.get("silence_boundary", True)) and envelope_getter is not None

        windows: list[tuple[float, float]] = []
        hard_flags: list[bool] = []
        start = start_from
        while start < total_sec - 1e-3:
            if total_sec - start <= max_sec + 1e-3:
                windows.append((start, total_sec))
                hard_flags.append(False)
                break
            target = start + max_sec
            band_lo = max(start + 5.0, target - lookback)
            band_hi = min(target, total_sec - 0.5)
            cut, hard = target, True
            if use_silence and band_hi > band_lo + 1e-3:
                rms = self._load_band(envelope_getter, band_lo, band_hi)
                if rms is not None:
                    found = find_silence_cut(
                        rms, band_lo, target=target, silence_min_sec=silence_min,
                    )
                    if (found is not None and band_lo <= found <= band_hi
                            and found >= start + 5.0):
                        cut, hard = float(found), False
            windows.append((start, cut))
            hard_flags.append(hard)
            start = cut - overlap
        return windows, hard_flags

    @staticmethod
    def _load_band(getter, lo: float, hi: float):
        """容错调用包络取数器（任何异常 → None，回退硬切）。"""
        try:
            return getter(float(lo), float(hi))
        except Exception:
            return None

    @staticmethod
    def _as_float(value, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _make_window_clip(self, audio_path, task, idx: int,
                          start: float, end: float) -> Path:
        """ffmpeg 切出窗口级 16kHz 单声道 WAV（供模型窗口内解码）。

        输出到项目 temp 目录（任务 id 命名），整个任务结束后由调用方清理。
        """
        from app.ffmpeg import run_ffmpeg

        tid = str(getattr(task, "id", None) or "moss_task")
        safe_tid = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in tid)
        out = (project_root / "temp" / "moss_windows" / safe_tid
               / f"win_{idx:03d}_{int(start):06d}.wav")
        out.parent.mkdir(parents=True, exist_ok=True)
        duration = max(float(end) - float(start), 0.1)
        proc = run_ffmpeg(
            [
                "-hide_banner", "-loglevel", "error", "-y",
                "-ss", f"{float(start):.3f}",
                "-i", str(audio_path),
                "-t", f"{duration:.3f}",
                "-vn", "-ac", "1", "-ar", "16000",
                "-c:a", "pcm_s16le",
                str(out),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.returncode != 0:
            tail = "\n".join((proc.stderr or "").strip().splitlines()[-3:])
            raise RuntimeError(
                f"长音频分窗切片失败（{float(start):.1f}-{float(end):.1f}s）: {tail}"
            )
        return out

    def _window_max_new_tokens(self, base, window_sec: float) -> int:
        """窗口级生成预算：与用户上限取小者，避免整段预算失控生成。

        默认整段预算 65536 是为超长音频准备的；分窗后每窗最多
        16 token/s 已覆盖密集语音 + 时间戳/说话人开销。
        """
        budget = max(1024, int(float(window_sec) * _WINDOW_TOKEN_RATE) + 256)
        try:
            base = int(base or 0)
        except (TypeError, ValueError):
            base = 0
        if base <= 0:
            return budget
        return max(1, min(base, budget))

    def _transcribe_windowed(self, *, task, audio_path, windows, hard_flags,
                             total_sec, merged, base_kwargs, state, started,
                             check_cancel, emit_window_end, envelope_getter=None,
                             window_sec=None) -> tuple[list, dict]:
        """逐窗转写 → 时间轴平移 → 硬切修复 + 去重合并 → 合成 info。

        流水线（GPU 优化方案 A）：后台线程并行执行窗口的切窗 + 解码/mel/
        tokenize（``prepare_clip``，无模型锁），主线程执行 ``generate_with``
        （持模型锁）——窗口 i 的推理与窗口 i+1 的输入准备重叠，消除
        ffmpeg 切片/特征提取等 CPU 开销造成的 GPU 空闲期。

        运行时 OOM 退避：单窗仍超出显存时，缩窗 ×0.7（下限 45s）
        重规划**剩余**音频（已完成窗口保留），重建预取队列与线程后重试，
        最多 3 次；退避耗尽后抛出带指引的错误。
        """
        import queue as _queue
        import threading

        window_segments: list[list[dict]] = []
        infos: list[dict] = []
        clip_dir: Optional[Path] = None
        effective_sec = float(window_sec or max(
            (float(e) - float(s) for s, e in windows), default=DEFAULT_WINDOW_SEC,
        ))
        retreats = 0
        stop_event = threading.Event()
        ready = _queue.Queue(maxsize=2)  # 最多预取 2 个窗口

        prepare_prompt = base_kwargs.get("prompt")
        prepare_max_length = int(base_kwargs.get("max_length", 131072))

        def _prepare_worker(window_list: list, start_idx: int) -> None:
            """后台线程：切窗 + prepare_clip（无模型锁），与主线程 generate 并行。

            队列满时以超时循环等待（0.2s 轮询 stop_event），取消后能退出，
            不会因阻塞在 put 而永久持有 GPU 输入张量。
            """
            for widx in range(start_idx, len(window_list)):
                if stop_event.is_set():
                    return
                wstart, wend = window_list[widx]
                try:
                    clip = self._make_window_clip(audio_path, task, widx, wstart, wend)
                    inputs, prompt_len = self.runner.prepare_clip(
                        clip, prompt=prepare_prompt, max_length=prepare_max_length,
                    )
                except Exception as exc:
                    while not stop_event.is_set():
                        try:
                            ready.put((widx, None, exc, None), timeout=0.2)
                            return
                        except _queue.Full:
                            continue
                while not stop_event.is_set():
                    try:
                        ready.put((widx, inputs, prompt_len, clip), timeout=0.2)
                        break
                    except _queue.Full:
                        continue

        def _start_worker(window_list: list, start_idx: int) -> None:
            stop_event.clear()
            threading.Thread(
                target=_prepare_worker, args=(window_list, start_idx), daemon=True
            ).start()

        _start_worker(windows, 0)
        try:
            idx = 0
            while idx < len(windows):
                check_cancel()
                widx, inputs, prompt_len, clip = ready.get()
                if inputs is None:
                    raise prompt_len  # 上游 prepare 失败，payload 为异常对象
                if clip_dir is None:
                    clip_dir = clip.parent
                start, end = windows[idx]
                state["pos_base"] = float(start)
                kwargs = dict(base_kwargs)
                kwargs["max_new_tokens"] = self._window_max_new_tokens(
                    kwargs.get("max_new_tokens"), end - start,
                )
                generate_kwargs = {
                    "max_new_tokens": kwargs.get("max_new_tokens"),
                    "do_sample": str(kwargs.get("decoding", "greedy")) == "sample",
                    "temperature": kwargs.get("temperature"),
                    "top_p": kwargs.get("top_p"),
                    "top_k": kwargs.get("top_k"),
                    "status_callback": kwargs.get("status_callback"),
                    "partial_text_callback": kwargs.get("partial_text_callback"),
                }
                try:
                    result = self.runner.generate_with(
                        inputs, prompt_len, **generate_kwargs
                    )
                except Exception as exc:
                    if isinstance(exc, CancelledError):
                        raise  # 任务取消不是转写失败，不记 error
                    if "out of memory" not in str(exc).lower():
                        self._log(
                            "error",
                            f"[transcribe] 长音频窗口 {idx + 1}/{len(windows)} 转写失败"
                            f"（{float(start):.1f}-{float(end):.1f}s）: {exc}",
                        )
                        raise
                    # ── 运行时 OOM 退避：停预取、缩窗重规划、重建队列与线程 ──
                    retreats += 1
                    if retreats > _OOM_MAX_RETREATS or effective_sec <= _OOM_FLOOR_SEC + 1e-3:
                        self._log(
                            "error",
                            f"[transcribe] 长音频窗口 {float(start):.1f}-{float(end):.1f}s "
                            f"连续 OOM（退避 {retreats - 1} 次后单窗已到 {_OOM_FLOOR_SEC:.0f}s 下限），"
                            "请把 MOSS 服务 device 改为 cpu，或释放显存后重试",
                        )
                        raise
                    stop_event.set()
                    effective_sec = max(_OOM_FLOOR_SEC, effective_sec * _OOM_SHRINK)
                    tail, tail_flags = self._plan_windows_with_flags(
                        total_sec, merged, envelope_getter,
                        window_sec=effective_sec, start_from=float(start),
                    )
                    if not tail:
                        raise
                    self._empty_cuda_cache()
                    self._log(
                        "warning",
                        f"[transcribe] 窗口 {float(start):.1f}-{float(end):.1f}s OOM："
                        f"单窗缩至 {effective_sec:.0f}s，剩余音频重规划为 "
                        f"{len(tail)} 个窗口后重试",
                    )
                    windows = windows[:idx] + tail
                    hard_flags = hard_flags[:idx] + tail_flags
                    ready = _queue.Queue(maxsize=2)  # 丢弃旧预取，重建队列
                    _start_worker(windows, idx)      # 从当前窗口重启预取
                    continue
                self._note_first_load(started)
                text = getattr(result, "text", "")
                info = result.to_dict() if hasattr(result, "to_dict") else dict(result or {})
                if not isinstance(info, dict):
                    info = {}
                # 窗口内先不做单说话人合并（跨窗口边界由最终合并统一处理）
                segs = self._segments_from_text(
                    text, merged, offset=float(start), apply_single=False,
                )
                window_segments.append(segs)
                infos.append(info)
                state["tokens_base"] = int(state.get("tokens_base") or 0) + int(
                    info.get("generated_tokens") or 0
                )
                state["tokens"] = 0
                self._log(
                    "info",
                    f"[transcribe] 长音频窗口 {idx + 1}/{len(windows)} 完成"
                    f"（{float(start):.1f}-{float(end):.1f}s，用时="
                    f"{time.time() - started:.1f}s 累计）",
                )
                emit_window_end(end)
                idx += 1
            seg_dicts = self._merge_window_segments(
                window_segments, windows, merged, hard_flags=hard_flags,
            )
        finally:
            stop_event.set()  # 停止预取线程（daemon，不阻塞退出）
            # 排空预取队列，释放 GPU 输入张量引用（取消路径防显存泄漏）
            while True:
                try:
                    _widx, _inputs, _plen, _clip = ready.get_nowait()
                    if _inputs is not None:
                        del _inputs
                except _queue.Empty:
                    break
            if clip_dir is not None:
                shutil.rmtree(clip_dir, ignore_errors=True)
            self._empty_cuda_cache()  # 取消/OOM/正常结束统一清 allocator 缓存
        info = self._combine_window_info(
            audio_path, windows, infos, seg_dicts,
            hard_flags=hard_flags, window_sec=effective_sec,
        )
        return seg_dicts, info

    @staticmethod
    def _empty_cuda_cache():
        """OOM 退避时清空 CUDA 分配器缓存（失败静默）。"""
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    def _merge_window_segments(self, window_segments: list, windows: list,
                               merged: dict, hard_flags: Optional[list] = None) -> list:
        """窗口段合并：硬切修复 → 重叠丢弃/裁剪 → 残余重叠消解 → 归一。

        规则（与分窗策略配套）：
        - 硬切边界先做文本修复（``_repair_boundary_cuts``）：被切点截断的
          句尾段与下一窗口续接的句头段缝合为完整段（带 ``_repaired`` 标记，
          跳过后续起点裁剪，保留真实起点）；
        - 第 0 窗全保留；
        - 第 k>0 窗：end ≤ 前一窗末尾（即完全落在重叠区）的段丢弃；
          start < 前一窗末尾但 end 越过边界的段，start 裁剪到边界；
        - 最终按时间轴解析残余重叠（不同窗口时间戳抖动）：先到者优先，
          后者完全被覆盖或文本高度相似时丢弃，否则把起点裁到前段末尾。
        - single_speaker=true 时对合并结果统一归一（合并/排序后执行）。
        """
        if hard_flags:
            window_segments = self._repair_boundary_cuts(
                window_segments, windows, hard_flags,
            )
        out: list[dict] = []
        for idx, (_wstart, _wend) in enumerate(windows):
            segs = window_segments[idx] if idx < len(window_segments) else []
            for seg in segs:
                seg = dict(seg)
                repaired = bool(seg.pop("_repaired", False))
                if idx > 0 and not repaired:
                    prev_end = float(windows[idx - 1][1])
                    if float(seg.get("end", 0.0)) <= prev_end + 0.01:
                        continue
                    if float(seg.get("start", 0.0)) < prev_end:
                        seg["start"] = prev_end
                if float(seg.get("end", 0.0)) > float(seg.get("start", 0.0)):
                    out.append(seg)
        out = self._resolve_segment_overlaps(out)
        if merged.get("single_speaker", False):
            out = force_single_speaker(out)
        out.sort(key=lambda s: (float(s.get("start", 0.0)), float(s.get("end", 0.0))))
        for i, seg in enumerate(out, 1):
            seg["id"] = f"seg_{i:04d}"
        return out

    def _repair_boundary_cuts(self, window_segments: list, windows: list,
                              hard_flags: list) -> list:
        """硬切边界文本修复：截断句尾 + 续接句头 → 缝合为完整段。

        仅处理 hard_flag=True 的边界（切点未落在静音里，语音可能被
        截断）。配对条件：上一窗口存在段尾距切点 ≤1.2s 的句尾段，且
        下一窗口存在起点 ≈ 切点、终点越过切点的句头段（同一句话的
        续接），二者文本相似度达标（截断句尾的后缀出现在句头前部，
        或整体相似度 ≥0.5）。缝合文本经前后缀去重拼接，时间段取
        [句尾起点, 句头终点]；句头段从下一窗口移除，避免双计。
        """
        segs = [[dict(s) for s in w] for w in window_segments]
        for k in range(min(len(windows), len(segs)) - 1):
            if k >= len(hard_flags) or not hard_flags[k]:
                continue
            b = float(windows[k][1])
            ov = max(0.0, b - float(windows[k + 1][0]))
            tails = sorted(
                [
                    i for i, s in enumerate(segs[k])
                    if (float(s.get("start", 0.0)) < b
                        and float(s.get("end", 0.0)) >= b - _HARD_CUT_TAIL_TOL)
                ],
                key=lambda i: float(segs[k][i].get("end", 0.0)),
                reverse=True,
            )
            heads = sorted(
                [
                    j for j, s in enumerate(segs[k + 1])
                    if (float(s.get("start", 0.0)) <= b + _HARD_CUT_HEAD_TOL
                        and float(s.get("end", 0.0)) > b + _HARD_CUT_HEAD_TOL)
                ],
                key=lambda j: float(segs[k + 1][j].get("start", 0.0)),
            )
            used_heads = set()
            for ti in tails:
                tail = segs[k][ti]
                hj = None
                for j in heads:
                    if j in used_heads:
                        continue
                    if self._boundary_pair_score(tail, segs[k + 1][j]) > 0.0:
                        hj = j
                        break
                if hj is None:
                    continue
                head = segs[k + 1][hj]
                used_heads.add(hj)
                merged = dict(tail)
                merged["start"] = min(
                    float(tail.get("start", 0.0)), float(head.get("start", 0.0)),
                )
                merged["end"] = float(head.get("end", tail.get("end", b)))
                merged["text"] = self._join_texts(
                    tail.get("text", ""), head.get("text", ""),
                )
                merged["_repaired"] = True
                segs[k][ti] = merged
            if used_heads:
                segs[k + 1] = [
                    s for j, s in enumerate(segs[k + 1]) if j not in used_heads
                ]
        return segs

    def _boundary_pair_score(self, tail: dict, head: dict) -> float:
        """截断句尾/续接句头的配对置信度（0 = 不配对）。

        截断句尾的末尾字符应出现在句头文本的前部（窗口只覆盖句头段
        长度两倍的前缀）；否则退化为整体相似度 ≥0.5 才配对。
        """
        a = self._norm_text(tail.get("text", ""))
        b = self._norm_text(head.get("text", ""))
        if not a or not b:
            return 0.0
        prefix = b[: min(len(b), max(6, len(a) * 2))]
        # 截断句尾的末尾 1-3 字符应出现在句头文本前部（截断处常为半词）
        if len(a) >= 2:
            for length in (min(3, len(a)), 2):
                if a[-length:] in prefix:
                    return 1.0
        elif a in prefix:
            return 1.0
        if a in b or b in a:
            return 0.9
        try:
            ratio = difflib.SequenceMatcher(None, a, b).ratio()
        except Exception:
            return 0.0
        return ratio if ratio >= 0.5 else 0.0

    @staticmethod
    def _norm_text(text) -> str:
        return "".join(ch for ch in str(text or "") if ch.isalnum())

    @staticmethod
    def _join_texts(a: str, b: str) -> str:
        """两句文本按最长前后缀重叠去重拼接（无重叠则直接拼接）。"""
        a, b = (a or "").strip(), (b or "").strip()
        if not a:
            return b
        if not b:
            return a
        for length in range(min(len(a), len(b), 20), 0, -1):
            if a[-length:] == b[:length]:
                return a + b[length:]
        return a + b

    def _resolve_segment_overlaps(self, segments: list) -> list:
        """时间戳重叠消解（分窗边界专用）：先到者优先，防重复段/覆盖双计。"""
        segs = [dict(s) for s in segments if s]
        segs.sort(key=lambda s: (float(s.get("start", 0.0)), float(s.get("end", 0.0))))
        resolved: list[dict] = []
        for seg in segs:
            if resolved:
                prev = resolved[-1]
                if float(seg.get("start", 0.0)) < float(prev.get("end", 0.0)) - 0.02:
                    if (float(seg.get("end", 0.0)) <= float(prev.get("end", 0.0)) + 0.02
                            or self._same_utterance(seg.get("text", ""), prev.get("text", ""))):
                        continue
                    seg["start"] = float(prev["end"])
            if float(seg.get("end", 0.0)) - float(seg.get("start", 0.0)) > 0.02:
                resolved.append(seg)
        return resolved

    @staticmethod
    def _same_utterance(text_a, text_b) -> bool:
        """边界去重用的文本相似判定（包含关系或高相似度）。"""
        def _norm(text) -> str:
            return "".join(ch for ch in str(text or "") if ch.isalnum())

        a, b = _norm(text_a), _norm(text_b)
        if not a or not b:
            return False
        if a in b or b in a:
            return True
        try:
            return difflib.SequenceMatcher(None, a, b).ratio() >= 0.85
        except Exception:
            return False

    def _combine_window_info(self, audio_path, windows: list,
                             infos: list, seg_dicts: list,
                             hard_flags: Optional[list] = None,
                             window_sec: Optional[float] = None) -> dict:
        """多窗口 TranscriptionResult.to_dict() 合成（token/耗时求和）。"""
        base = dict(infos[0]) if infos else {}
        base.update({
            "text": self._segments_to_transcript_text(seg_dicts),
            "generated_tokens": sum(int(i.get("generated_tokens") or 0) for i in infos),
            "elapsed_sec": round(
                sum(float(i.get("elapsed_sec") or 0.0) for i in infos), 3
            ),
            "audio": str(audio_path),
            "windows": len(windows),
            "chunking": {
                "strategy": "sliding_window_silence_aware",
                "window_sec": round(float(window_sec), 1) if window_sec else None,
                "silence_boundaries": sum(1 for f in (hard_flags or []) if not f),
                "hard_boundaries": sum(1 for f in (hard_flags or []) if f),
            },
        })
        return base

    @staticmethod
    def _segments_to_transcript_text(seg_dicts: list) -> str:
        """段 dict → 标准 LRC 转录文本（[mm:ss.cs]<说话人>正文，info.text 兼容展示）。"""
        parts = []
        for s in seg_dicts or []:
            speaker = str(s.get("speaker") or "S01")
            parts.append(
                f"[{format_lrc_time(float(s.get('start', 0.0)))}]<{speaker}>"
                f"{s.get('text', '')}"
            )
        return "\n".join(parts)

    def _build_transcribe_kwargs(self, merged: dict, hotwords) -> dict:
        """按参数类别装配 runner.transcribe(**kwargs)。"""
        decoding = merged.get("decoding", "greedy")
        transcribe_kwargs = {
            "max_length": int(merged.get("max_len", 131072)),
            "max_new_tokens": int(merged.get("max_new_tokens", 65536)),
            "decoding": decoding,
        }
        # sampling 参数仅 sample 解码时生效（greedy 传了也会被 runner 忽略，
        # 此处干脆不传，与 vendor CLI 行为一致）
        if decoding == "sample":
            for key, cast in (("temperature", float), ("top_p", float), ("top_k", int)):
                value = merged.get(key)
                if value not in (None, ""):
                    transcribe_kwargs[key] = cast(value)
        # 热词：按官方配方附加到提示词末尾（无显式 prompt 时用 vendor 默认提示词兜底）
        prompt = merged.get("prompt")
        hotword_text = self._normalize_hotwords(hotwords)
        if hotword_text:
            if not prompt:
                from moss_transcribe_diarize.inference_utils import DEFAULT_PROMPT
                prompt = DEFAULT_PROMPT
            prompt = f"{prompt}\n热词提示：{hotword_text}"
        if prompt:
            transcribe_kwargs["prompt"] = prompt
        return transcribe_kwargs

    def _note_first_load(self, started: float):
        """首次转写完成后补记真实设备（ModelRunner 懒加载）。"""
        if self._device_logged:
            return
        self._device_logged = True
        device = getattr(self.runner, "_device", None)
        dtype = getattr(self.runner, "_dtype", None)
        self._log(
            "info",
            f"MOSS 首次转写完成（模型实际加载，首窗/首段用时={time.time() - started:.1f}s；"
            f"分窗模式后续窗口逐窗另记）：实际设备={device}，dtype={dtype}，",
        )
        if self.on_first_load is not None:
            try:
                self.on_first_load(device, dtype)
            except Exception:
                pass  # 回调失败不影响转写结果

    def _segments_from_text(self, text: str, merged: dict,
                            offset: float = 0.0, apply_single: bool = True) -> list:
        """原生紧凑转录 → 字幕段列表（完成结果与实时预览共用）。

        postprocess=False 保留模型原始切分，说话人归一化交给
        force_single_speaker 兜底；文本为空或解析失败时返回空列表。
        ``offset`` 用于分窗段回挂全局时间轴。
        """
        if not text:
            return []
        try:
            from moss_transcribe_diarize.subtitle import subtitle_segments_from_transcript

            segments = subtitle_segments_from_transcript(text, postprocess=False)
            seg_dicts = [s.to_dict() for s in segments]
        except Exception as exc:
            self._log("warning", f"MOSS 段切分失败（跳过本次刷新）: {exc}")
            return []
        if offset:
            for seg in seg_dicts:
                seg["start"] = float(seg.get("start", 0.0)) + float(offset)
                seg["end"] = float(seg.get("end", 0.0)) + float(offset)
        if apply_single and merged.get("single_speaker", False):
            seg_dicts = force_single_speaker(seg_dicts)
        return seg_dicts

    @staticmethod
    def _normalize_hotwords(hotwords) -> str:
        """热词归一为逗号串（dict {"hotwords": [...]} / list / str 均可）。"""
        if not hotwords:
            return ""
        if isinstance(hotwords, dict):
            hotwords = hotwords.get("hotwords", [])
        if isinstance(hotwords, list):
            return ",".join(str(h) for h in hotwords if str(h).strip())
        return str(hotwords).strip()

    def _resolve_task(self, task):
        """转写语义解析：file_path 作为音频路径，configs["args"] 提供参数，
        configs["hotwords"] 提供热词（选「无」时为 None）。"""
        _source, configs = super()._resolve_task(task)
        args = configs.get("args")
        return {
            "audio_path": str(task.file_path),
            "transcribe_config": args if isinstance(args, dict) else None,
            "hotwords": configs.get("hotwords"),
        }

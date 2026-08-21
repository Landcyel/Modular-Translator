"""MOSS 流式转写 Runner —— 不改 vendor 源码的部分文本预览方案。

vendor 的 ``ModelRunner.transcribe`` 内部使用只数 token 的
``ProgressStreamer``，运行中拿不到部分文本。本模块通过**子类覆盖**
提供能力：

- 复用父类 ``_lock`` / ``_ensure_loaded`` / ``_device`` / ``_dtype``；
- 复用 vendor 公开函数 ``build_transcription_messages`` / ``prepare_inputs``
  与 ``generation_progress`` / ``TranscriptionResult``；
- 唯一差异：streamer 换成 ``PartialTextStreamer``，在计数 token 的同时
  节流解码部分文本并回调。

生成调用序列与 vendor ``generate_transcription`` 完全一致（autocast、
inference_mode、TypeError 去 streamer 重试兜底），vendor 源码零修改。
"""
from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Callable, Optional

import torch
from transformers.generation.streamers import BaseStreamer

from moss_transcribe_diarize.app.model_runner import (
    ModelRunner,
    TranscriptionResult,
)
from moss_transcribe_diarize.inference_utils import (
    DEFAULT_PROMPT,
    build_transcription_messages,
    prepare_inputs,
)

# 与 vendor inference_utils._token_count 保持一致（不依赖私有符号）
def _token_count(value) -> int:
    if hasattr(value, "numel"):
        return int(value.numel())
    if isinstance(value, (list, tuple)):
        return sum(_token_count(item) for item in value)
    return 1


class PartialTextStreamer(BaseStreamer):
    """token 计数 + 部分文本节流解码的 HF streamer。

    与 vendor ``ProgressStreamer`` 的语义对齐：第一次 ``put`` 视为
    prompt prefill 直接跳过，之后累积生成 token ids。部分文本只在
    ``min_tokens`` 或 ``min_interval`` 阈值触发时解码回调，避免逐
    token 解码与 UI 逐 token 重建。
    """

    def __init__(
        self,
        tokenizer,
        token_callback: Optional[Callable[[int], None]] = None,
        partial_text_callback: Optional[Callable[[str, int], None]] = None,
        *,
        min_interval: float = 0.25,
        min_tokens: int = 16,
        sync_batch: int = 8,
    ):
        self.tokenizer = tokenizer
        self.token_callback = token_callback
        self.partial_text_callback = partial_text_callback
        self.min_interval = min_interval
        self.min_tokens = max(1, int(min_tokens))
        self.sync_batch = max(1, int(sync_batch))

        self.generated_tokens = 0
        self._seen_prompt = False
        self._ids: list[torch.Tensor] = []
        # 待同步的 GPU token（批量 .cpu()，避免每 token 一次 GPU→CPU 同步）
        self._pending_ids: list[torch.Tensor] = []
        self._pending_count = 0
        self._last_emit_tokens = 0
        self._last_emit_time = time.monotonic()

    def put(self, value):
        count = _token_count(value)
        if not self._seen_prompt:
            # 与 vendor ProgressStreamer 一致：第一次 put 为 prompt prefill
            self._seen_prompt = True
            return
        self.generated_tokens += count
        self._collect_ids(value)

        if self.token_callback is not None:
            self.token_callback(self.generated_tokens)

        if self.partial_text_callback is not None and self._should_emit():
            text = self._decode()
            if text:
                self.partial_text_callback(text, self.generated_tokens)

    def end(self):
        return None

    def _collect_ids(self, value):
        """累积 GPU token，攒满 sync_batch 才批量 .cpu()（降 8 倍同步频率）。

        仅当值累计达到阈值或显式 flush 时才触发同步，避免打断 GPU 异步流水线。
        """
        if isinstance(value, torch.Tensor) and value.numel():
            self._pending_ids.append(value.detach().reshape(-1))
            self._pending_count += value.numel()
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, torch.Tensor) and item.numel():
                    self._pending_ids.append(item.detach().reshape(-1))
                    self._pending_count += item.numel()
        if self._pending_count >= self.sync_batch:
            self._flush_pending()

    def _flush_pending(self):
        """把累积的 GPU token 批量取回 CPU 并追加到已解码 id 列表。"""
        if not self._pending_ids:
            return
        self._ids.append(torch.cat(self._pending_ids).cpu())
        self._pending_ids = []
        self._pending_count = 0

    def _should_emit(self) -> bool:
        now = time.monotonic()
        if (self.generated_tokens - self._last_emit_tokens) >= self.min_tokens:
            self._last_emit_tokens = self.generated_tokens
            self._last_emit_time = now
            return True
        if (now - self._last_emit_time) >= self.min_interval:
            self._last_emit_tokens = self.generated_tokens
            self._last_emit_time = now
            return True
        return False

    def _decode(self) -> str:
        self._flush_pending()  # 生成结束前把剩余 < sync_batch 的 token 也取回
        if not self._ids:
            return ""
        try:
            ids = torch.cat(self._ids)
            return self.tokenizer.decode(
                ids.tolist(), skip_special_tokens=True
            ).strip()
        except Exception:
            return ""


class StreamingModelRunner(ModelRunner):
    """支持 ``partial_text_callback`` 的 ModelRunner 子类。

    ``SUPPORTS_PARTIAL_TEXT = True`` 供执行器探测能力；老版本 vendor 或
    自定义 streamer 不可用时，调用方应直接使用 vendor ``ModelRunner``。
    """

    SUPPORTS_PARTIAL_TEXT = True

    def transcribe(
        self,
        audio_path: str | Path,
        *,
        prompt: str = DEFAULT_PROMPT,
        max_length: int = 131072,
        max_new_tokens: int = 2048,
        decoding: str = "greedy",
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        status_callback: Optional[Callable] = None,
        partial_text_callback: Optional[Callable[[str, int], None]] = None,
    ) -> TranscriptionResult:
        if partial_text_callback is None:
            return super().transcribe(
                audio_path,
                prompt=prompt,
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                decoding=decoding,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                status_callback=status_callback,
            )
        return self._transcribe_streaming(
            audio_path,
            prompt=prompt,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            decoding=decoding,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            status_callback=status_callback,
            partial_text_callback=partial_text_callback,
        )

    def _transcribe_streaming(
        self,
        audio_path,
        *,
        prompt,
        max_length,
        max_new_tokens,
        decoding,
        temperature,
        top_p,
        top_k,
        status_callback,
        partial_text_callback,
    ) -> TranscriptionResult:
        do_sample = decoding == "sample"
        if status_callback is not None:
            status_callback("loading_model", 0.05, None)
        inputs, prompt_len = self.prepare_clip(
            audio_path, prompt=prompt, max_length=max_length,
        )
        if status_callback is not None:
            status_callback("transcribing", 0.25, None)
        return self.generate_with(
            inputs, prompt_len,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            status_callback=status_callback,
            partial_text_callback=partial_text_callback,
        )

    def prepare_clip(
        self,
        audio_path: str | Path,
        *,
        prompt: str = DEFAULT_PROMPT,
        max_length: int = 131072,
    ) -> tuple[dict, int]:
        """窗口/单文件转写的输入准备：波形解码 + mel 特征 + tokenize（无模型锁）。

        可在后台线程与其它窗口的 ``generate_with`` 并行执行（流水线），
        把 ffmpeg 切片/解码/特征提取等开销与 GPU 推理重叠。懒加载且设备
        未确定时回退到持锁加载（仅首窗串行，流水线从第二窗开始生效）。
        """
        if getattr(self, "_device", None) is None:
            with self._lock:
                self._ensure_loaded()
        # 与 vendor generate_transcription 相同的 autocast 策略
        context = (
            torch.amp.autocast("cuda", dtype=self._dtype)
            if self._device.type == "cuda"
            and self._dtype in (torch.float16, torch.bfloat16)
            else torch.no_grad()
        )
        with context:
            inputs = prepare_inputs(
                self._processor,
                build_transcription_messages(audio_path, prompt),
                max_length=max_length,
                device=self._device,
            ).to(self._device)
        prompt_len = int(inputs["attention_mask"][0].sum().item())
        return inputs, prompt_len

    def generate_with(
        self,
        inputs: dict,
        prompt_len: int,
        *,
        max_new_tokens: int = 2048,
        do_sample: bool = False,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        status_callback: Optional[Callable] = None,
        partial_text_callback: Optional[Callable[[str, int], None]] = None,
    ) -> TranscriptionResult:
        """纯 generate（持模型锁），与 ``prepare_clip`` 配对使用。

        长音频流水线：窗口 i 的 generate 与窗口 i+1 的 prepare 并行。
        """
        with self._lock:
            if status_callback is not None:
                status_callback("loading_model", 0.05, None)
            self._ensure_loaded()
            if status_callback is not None:
                status_callback("transcribing", 0.10, None)

            def on_partial_text(partial_text: str, generated_tokens: int) -> None:
                if partial_text_callback is not None:
                    partial_text_callback(partial_text, generated_tokens)

            # 进度只由部分文本（已确认段）驱动：不注册 token 级回调，
            # 避免 token 级无段发射把进度打回 0% 造成闪烁。
            streamer = PartialTextStreamer(
                self._processor.tokenizer,
                partial_text_callback=on_partial_text,
            )

            started = time.time()

            generation_config = copy.deepcopy(self._model.generation_config)
            if max_new_tokens is not None:
                generation_config.max_new_tokens = max_new_tokens
            generation_config.do_sample = do_sample
            if do_sample and temperature is not None:
                generation_config.temperature = temperature
            if do_sample and top_p is not None:
                generation_config.top_p = top_p
            if do_sample and top_k is not None:
                generation_config.top_k = top_k

            generate_kwargs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "input_features": inputs["input_features"],
                "audio_feature_lengths": inputs["audio_feature_lengths"],
                "audio_chunk_mapping": inputs["audio_chunk_mapping"],
                "generation_config": generation_config,
                "streamer": streamer,
            }

            with torch.inference_mode(), (
                torch.amp.autocast("cuda", dtype=self._dtype)
                if self._device.type == "cuda"
                and self._dtype in (torch.float16, torch.bfloat16)
                else torch.no_grad()
            ):
                try:
                    outputs = self._model.generate(**generate_kwargs)
                except TypeError as exc:
                    # 与 vendor 相同：部分模型 generate 不接受 streamer
                    if "streamer" not in str(exc):
                        raise
                    generate_kwargs.pop("streamer", None)
                    outputs = self._model.generate(**generate_kwargs)

            generated_ids = outputs[0][prompt_len:]
            text = self._processor.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            ).strip()
            # 收尾回调：确保预览包含最终文本（部分模型 streamer 不触发末次 put）。
            # 进度满格由执行器 execute() 收尾统一补发，此处不再发状态回调。
            if partial_text_callback is not None:
                partial_text_callback(text, int(generated_ids.numel()))

            return TranscriptionResult(
                text=text,
                prompt_len=prompt_len,
                generated_tokens=int(generated_ids.numel()),
                elapsed_sec=time.time() - started,
                model=self.model_path,
                audio="<prepared>",
                decoding="greedy" if not do_sample else "sample",
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                top_k=top_k if do_sample else None,
            )


__all__ = ["StreamingModelRunner", "PartialTextStreamer"]

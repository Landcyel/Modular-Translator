"""MOSS streaming transcription Runner — partial-text preview without modifying vendor source.

The vendor's ``ModelRunner.transcribe`` internally uses a token-counting
``ProgressStreamer``, so partial text is unavailable mid-run. This module provides
the capability via **subclass override**:

- Reuses the parent's ``_lock`` / ``_ensure_loaded`` / ``_device`` / ``_dtype``;
- Reuses the vendor public functions ``build_transcription_messages`` / ``prepare_inputs``
  and ``generation_progress`` / ``TranscriptionResult``;
- The only difference: the streamer is swapped for ``PartialTextStreamer``, which
  throttled-decodes partial text and calls back while counting tokens.

The generation call sequence is identical to the vendor ``generate_transcription``
(autocast, inference_mode, TypeError-retry fallback that drops the streamer), with
zero vendor source changes.
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

# Mirrors vendor inference_utils._token_count (without depending on the private symbol)
def _token_count(value) -> int:
    if hasattr(value, "numel"):
        return int(value.numel())
    if isinstance(value, (list, tuple)):
        return sum(_token_count(item) for item in value)
    return 1


class PartialTextStreamer(BaseStreamer):
    """HF streamer doing token counting + throttled partial-text decoding.

    Aligned with the vendor ``ProgressStreamer`` semantics: the first ``put`` is
    treated as prompt prefill and skipped; afterwards generated token ids accumulate.
    Partial text is decoded and reported only when the ``min_tokens`` or
    ``min_interval`` threshold triggers, avoiding per-token decoding and per-token UI rebuilds.
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
        # GPU tokens pending sync (batched .cpu(), avoiding one GPU→CPU sync per token)
        self._pending_ids: list[torch.Tensor] = []
        self._pending_count = 0
        self._last_emit_tokens = 0
        self._last_emit_time = time.monotonic()

    def put(self, value):
        count = _token_count(value)
        if not self._seen_prompt:
            # Same as vendor ProgressStreamer: the first put is the prompt prefill
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
        """Accumulate GPU tokens, batching .cpu() once sync_batch is reached (8× fewer syncs).

        Sync only triggers when the accumulated count reaches the threshold or an explicit
        flush, avoiding interruptions to the GPU async pipeline.
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
        """Batch-move accumulated GPU tokens back to CPU and append to the decoded id list."""
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
        self._flush_pending()  # before decode ends, also pull back remaining < sync_batch tokens
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
    """ModelRunner subclass supporting ``partial_text_callback``.

    ``SUPPORTS_PARTIAL_TEXT = True`` lets the executor detect the capability; when the
    older vendor or the custom streamer is unavailable, callers should use the vendor
    ``ModelRunner`` directly.
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
        """Input preparation for window/single-file transcription: waveform decode + mel features + tokenize (no model lock).

        Can run in a background thread in parallel with other windows' ``generate_with``
        (pipeline), overlapping ffmpeg cutting/decode/feature-extraction overhead with GPU
        inference. When lazy-loaded and the device is not yet resolved, falls back to
        lock-held loading (only the first window is serial; the pipeline takes effect from the second).
        """
        if getattr(self, "_device", None) is None:
            with self._lock:
                self._ensure_loaded()
        # Same autocast strategy as vendor generate_transcription
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
        """Pure generate (holds the model lock), paired with ``prepare_clip``.

        Long-audio pipeline: window i's generate runs in parallel with window i+1's prepare.
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

            # Progress is driven only by partial text (confirmed segments): no token-level
            # callback registered, avoiding token-level no-segment emissions pushing progress back to 0% (flicker).
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
                    # Same as vendor: some models' generate does not accept a streamer
                    if "streamer" not in str(exc):
                        raise
                    generate_kwargs.pop("streamer", None)
                    outputs = self._model.generate(**generate_kwargs)

            generated_ids = outputs[0][prompt_len:]
            text = self._processor.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            ).strip()
            # Final callback: ensure the preview includes the final text (some models' streamer
            # never triggers the last put). Full progress is emitted uniformly by the executor's
            # execute() wrap-up, so no status callback is sent here.
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

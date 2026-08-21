"""MOSS 音频时长探测工具（仅读取容器元数据，不解码音频）。

MOSS 白名单格式（mp3/wav/m4a/flac/ogg/opus）均可用 PyAV 解析；
PyAV 是 MOSS vendor 的既有依赖，无需新增安装项。
"""
from __future__ import annotations

from typing import Optional


def probe_duration(audio_path) -> Optional[float]:
    """返回音频时长（秒）；无法探测时返回 None（调用方回退比例进度）。

    依次尝试：音频流 duration → 帧数/采样率 → 容器 duration。
    """
    if not audio_path:
        return None
    try:
        import av  # 延迟导入：仅转写任务首次探测时加载

        with av.open(str(audio_path)) as container:
            stream = next(
                (s for s in container.streams if s.type == "audio"), None
            )
            if stream is None:
                return None
            # 流级 duration（单位：stream.time_base，秒）
            if stream.duration is not None:
                duration = float(stream.duration * stream.time_base)
                if duration > 0:
                    return duration
            # 帧数 / 采样率
            if stream.frames and stream.rate:
                duration = float(stream.frames) / float(stream.rate)
                if duration > 0:
                    return duration
            # 容器级 duration（单位：av.time_base，微秒）
            if container.duration is not None:
                duration = float(container.duration / av.time_base)
                if duration > 0:
                    return duration
    except Exception as exc:
        print(f"[moss] 音频时长探测失败 {audio_path}: {exc}")
    return None

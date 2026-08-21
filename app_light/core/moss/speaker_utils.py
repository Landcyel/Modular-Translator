"""单说话人后处理工具（MOSS 过度分离的兜底保险）。

纯 Python 实现、零外部依赖：主环境与独立环境均可直接导入。
"""
from __future__ import annotations


def force_single_speaker(segments, speaker: str = "S01", merge_gap: float = 0.3) -> list[dict]:
    """把所有段落统一为单一说话人，并将时间间隔 ≤ merge_gap 秒的相邻段合并。

    - ``segments``: ``[{"id", "start", "end", "speaker", "text"}, ...]`` 列表
    - ``speaker``: 统一使用的说话人标签（默认 ``S01``）
    - ``merge_gap``: 相邻段 ``start - prev.end`` 不超过该值（秒）则合并
      （text 拼接、end 取两段较大值）
    - 返回新列表（不改入参），按 start 排序

    与 MOSS 服务端 ``single_speaker`` 提示词配合形成"双保险"：提示词在模型
    生成侧抑制说话人分裂，本函数在结果侧兜底归一化。
    """
    segs = [
        {**s, "speaker": speaker}
        for s in segments
        if str(s.get("text", "")).strip()
    ]
    if not segs:
        return []
    segs.sort(key=lambda s: float(s["start"]))
    merged = [segs[0]]
    for s in segs[1:]:
        prev = merged[-1]
        if float(s["start"]) - float(prev["end"]) <= merge_gap:
            prev["text"] = (str(prev["text"]) + str(s["text"])).strip()
            prev["end"] = max(float(prev["end"]), float(s["end"]))
        else:
            merged.append(s)
    return merged

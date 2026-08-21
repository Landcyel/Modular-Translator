"""Single-speaker post-processing utilities (fallback insurance against MOSS over-segmentation).

Pure Python, zero external dependencies: importable from both the main and standalone environments.
"""
from __future__ import annotations


def force_single_speaker(segments, speaker: str = "S01", merge_gap: float = 0.3) -> list[dict]:
    """Force all segments to a single speaker and merge adjacent segments whose gap is ≤ merge_gap seconds.

    - ``segments``: a list of ``[{"id", "start", "end", "speaker", "text"}, ...]``
    - ``speaker``: the speaker label to use for all (default ``S01``)
    - ``merge_gap``: adjacent segments are merged when ``start - prev.end`` ≤ this value
      (seconds) (text concatenated, end takes the larger of the two)
    - Returns a new list (input unchanged), sorted by start

    Works with the MOSS server-side ``single_speaker`` prompt as a "double insurance":
    the prompt suppresses speaker splitting at generation time, while this function
    normalizes on the result side as a fallback.
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

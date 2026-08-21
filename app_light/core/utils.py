import json
from pathlib import Path

from app.paths import project_root as _PROJECT_ROOT

# Note: this module deliberately does not import numpy / inspect at top level.
# They are used only inside a few functions (test-audio generation), so they are
# lazily imported within those functions to avoid pulling heavy libs into the
# core chain when this module is imported, which would slow the app's first frame.


def load_json_file(file_path):
    """Load a JSON config file; accepts a path string, a Path object, or an already-parsed dict.

    Args:
        file_path: config file path (str or pathlib.Path) or an already-parsed config dict (dict).

    Returns:
        the parsed config dict.

    Raises:
        TypeError: when the argument is neither str, Path, nor dict.
    """
    if isinstance(file_path, dict):
        return file_path
    if isinstance(file_path, (str, Path)):
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    raise TypeError(f"load_json_file 需要 str、Path (路径) 或 dict，但收到了 {type(file_path).__name__}")


def load_noval_file(file_path):
    """Read a text file's content; also accepts already-loaded text content directly.

    Args:
        file_path: a file path (str) or already-loaded text content (str).
                   If it is an existing file path, its content is read; otherwise it is returned as-is.

    Returns:
        the file's text content.
    """
    if isinstance(file_path, str):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except FileNotFoundError:
            return file_path
    return file_path


def make_test_audio(duration: float = 5.0, sample_rate: int = 16000):
    import numpy as np

    t = np.arange(sample_rate * duration) / sample_rate
    audio = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    return audio


if __name__ == "__main__":
    pass

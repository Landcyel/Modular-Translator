import json
from pathlib import Path

from app.paths import project_root as _PROJECT_ROOT

# 注意：本模块故意不在顶层 import numpy / inspect。
# 它们只在少数函数内使用（测试音频生成），改为函数内惰性导入，
# 避免核心链导入本模块时连带加载重库、拖慢应用启动首帧。


def load_json_file(file_path):
    """加载 JSON 配置文件，支持路径字符串、Path 对象或已解析的 dict。

    Args:
        file_path: 配置文件路径 (str 或 pathlib.Path) 或已解析的配置字典 (dict)。

    Returns:
        解析后的配置字典。

    Raises:
        TypeError: 当参数既不是 str、Path 也不是 dict 时。
    """
    if isinstance(file_path, dict):
        return file_path
    if isinstance(file_path, (str, Path)):
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    raise TypeError(f"load_json_file 需要 str、Path (路径) 或 dict，但收到了 {type(file_path).__name__}")


def load_noval_file(file_path):
    """读取文本文件内容，也支持直接传入文本内容。

    Args:
        file_path: 文件路径 (str) 或已加载的文本内容 (str)。
                   若是存在的文件路径则读取其内容，否则直接返回。

    Returns:
        文件的文本内容。
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

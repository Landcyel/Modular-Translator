"""回归测试：窗口关闭 API 契约（flet 0.86.2）。

历史 bug：APP.py 曾用旧版 ``page.on_window_event`` + ``e.data == "close"``，
在 flet 0.86.2（重构版）下：
- ``Page`` 没有 ``on_window_event`` 属性（监听器改挂在 ``Window.on_event``），
  回调从未被注册 → 任务栏关闭被 ``prevent_close`` 拦截后无人调用 ``destroy()``，
  app 无法关闭；
- ``WindowEvent`` 用 ``type: WindowEventType`` 枚举字段承载事件类型，
  ``Event.data`` 不携带窗口事件类型。

本测试锁定 0.86.2 的窗口关闭相关 API 形状，防止旧版写法回归。
"""
import asyncio
import dataclasses

import flet as ft

from ui.layout import shutdown_and_destroy


def test_page_has_no_on_window_event():
    """Page 没有旧版 on_window_event 属性（0.86.2 重构后移除）。"""
    assert not hasattr(ft.Page, "on_window_event")


def test_window_on_event_exists():
    """窗口事件监听器挂在 Window.on_event（APP.py 现绑定处）。"""
    assert hasattr(ft.Window, "on_event")


def test_window_event_type_close_is_enum():
    """WindowEventType.CLOSE 是枚举成员，值为 "close"。"""
    assert isinstance(ft.WindowEventType.CLOSE, ft.WindowEventType)
    assert ft.WindowEventType.CLOSE.value == "close"


def test_window_event_uses_type_field():
    """WindowEvent 用 type 枚举字段承载事件类型（不是 Event.data）。"""
    field_names = {f.name for f in dataclasses.fields(ft.WindowEvent)}
    assert "type" in field_names


def test_window_prevent_close_and_destroy_exist():
    """prevent_close 属性、destroy()/close() 方法存在（APP.py 与顶栏按钮依赖）。"""
    assert hasattr(ft.Window, "prevent_close")
    assert hasattr(ft.Window, "destroy")
    assert hasattr(ft.Window, "close")


def test_page_window_field_is_window():
    """Page 声明 window 字段（APP.py 通过 page.window 访问）。"""
    assert "window" in ft.Page.__dataclass_fields__


def test_shutdown_and_destroy_orders_shutdown_before_destroy():
    """正常路径：shutdown 先于 destroy（防 llama-server 子进程残留）。"""
    calls = []

    class _Facade:
        def shutdown(self):
            calls.append("shutdown")

    class _Page:
        def __init__(self):
            self.window = _Window()

    class _Window:
        async def destroy(self):
            calls.append("destroy")

    asyncio.run(shutdown_and_destroy(_Facade(), _Page()))
    assert calls == ["shutdown", "destroy"]


def test_shutdown_and_destroy_destroys_even_if_shutdown_raises():
    """shutdown 异常不得阻断 destroy——prevent_close=True 下 destroy 是唯一出口，
    异常会导致窗口永远关不掉。"""
    calls = []

    class _Facade:
        def shutdown(self):
            calls.append("shutdown")
            raise RuntimeError("stop_service boom")

    class _Page:
        def __init__(self):
            self.window = _Window()

    class _Window:
        async def destroy(self):
            calls.append("destroy")

    asyncio.run(shutdown_and_destroy(_Facade(), _Page()))
    assert calls == ["shutdown", "destroy"]

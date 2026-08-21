"""回归测试：日志页显示方式（超长日志内容换行显示，不截断）。"""
from ui.pages.log import LogPage


def test_log_row_wraps_long_lines():
    """超长日志内容按区域宽度换行（no_wrap=False，无 ELLIPSIS 截断）。"""
    t = LogPage._row("[info] " + "x" * 500)
    assert t.no_wrap is False


def test_log_row_colors_by_level():
    """级别着色保留：error 红 / warn 橙（换行改动不影响）。"""
    from ui.theme import Palette

    err = LogPage._row("[error] x")
    warn = LogPage._row("[warn] x")
    info = LogPage._row("[info] x")
    assert err.color == Palette.ERROR
    assert warn.color == Palette.WARNING
    assert info.color == Palette.TEXT


class _FakePage:
    def run_task(self, coro):
        return None


def test_build_list_view_no_scroll_animation():
    """切换到日志页直接显示底部：auto_scroll 保持，但无滚动动画（duration 0 瞬时跳转）。

    flet 0.86.2：auto_scroll_animation 为 AnimationValue（True=1s 动画 / int 毫秒 /
    Animation），False 是非法值会使引擎端解析崩溃、子树渲染失败（全部灰色）；
    0 = duration 0 → jumpTo 瞬时跳转。
    """
    lp = LogPage(_FakePage())
    lp.build()
    assert lp.list_view.auto_scroll is True
    assert lp.list_view.auto_scroll_animation == 0

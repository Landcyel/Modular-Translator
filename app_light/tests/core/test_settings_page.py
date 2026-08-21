"""SettingsPage 页面层冒烟测试 — 无 GUI 会话（FakePage + 未挂载控件）。

覆盖：
1. build() 返回页面树（导航 + 编辑器 + 文件库组装）
2. _switch_form 切换配置类型并重建表单
3. _render_form 错误就地渲染
4. _duplicate_config 复制命名
5. form_values_from_refs 状态缓存
6. _save 无名称提示 / 校验失败提示 / 合法保存写盘（JSON 结构正确）
7. _safe_update 对未挂载控件不抛异常
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import flet as ft

from ui.pages.settings import SettingsPage


class FakeServices:
    def register_service(self, *a):
        pass


class FakePage:
    width = 1280
    _services = FakeServices()

    def run_task(self, coro_fn, *args):
        import asyncio
        asyncio.new_event_loop().run_until_complete(coro_fn(*args))

    def show_dialog(self, d):
        pass

    def update(self):
        pass


class _NameCtrl:
    """模拟 config_name_ref 控件（弱引用存活需强引用持有）。"""

    def __init__(self, value=""):
        self.value = value
        self.page = None

    def update(self):
        pass


def _make_page():
    sp = SettingsPage(FakePage())
    sp.build()
    return sp


def test_build_returns_page_tree():
    sp = _make_page()
    tree = sp.build()
    assert tree is not None
    # 顶栏“设置 · 配置工作台”已全局删除，build 直接返回工作台（宽屏 Row / 窄屏 Column）
    assert isinstance(tree, (ft.Row, ft.Column))
    assert sp.current_ct.key == "translate_default"  # 默认选中导航第一项（系统→翻译默认配置）


def test_save_without_name_prompts():
    sp = _make_page()
    # json 托管类型（如 prompt）无名称保存 → 提示输入名称
    sp._switch_form("prompt")
    sp._save(None)
    assert "请先输入配置名称" in sp.status_message
    assert sp.status_ok is False


def test_save_ini_default_writes_ini():
    sp = _make_page()
    # ini 托管类型（翻译默认配置）保存无需名称，直接写 default.ini
    sp._switch_form("translate_default")
    sp._save(None)
    assert "configs/system/default.ini" in sp.status_message
    assert sp.status_ok is True


def test_switch_form_rebuilds():
    sp = _make_page()
    sp._switch_form("rules")
    assert sp.current_ct.key == "rules"
    assert len(sp.current_ct.fields) == 5  # prefix/suffix/placeholder/recognize/skip（description 已移除）
    assert set(sp.field_refs) == {f.key for f in sp.current_ct.fields}


def test_render_form_with_errors():
    sp = _make_page()
    sp._switch_form("rules")
    sp._render_form(errors={"prefix": ["测试错误"]})
    assert sp.field_refs["prefix"] is not None


def test_duplicate_config_names_copy():
    sp = _make_page()
    name_ctrl = _NameCtrl("my_rules")
    # switch 会重建 save_row（含名称 TextField，ref 绑定到新控件），
    # 因此先切换再注入可控的名称控件
    sp._switch_form("rules")
    sp.config_name_ref.current = name_ctrl  # 需强引用存活（Ref 用 weakref）
    sp._duplicate_config(None)
    assert sp.config_name == "my_rules_copy"
    assert name_ctrl.value == "my_rules_copy"
    assert sp.status_ok is True


def test_state_cache_roundtrip():
    sp = _make_page()
    sp._switch_form("rules")
    cached = sp.current_ct.form_values_from_refs(sp.field_refs)
    assert set(cached) == {f.key for f in sp.current_ct.fields}
    # 缓存可直接 populate 回新表单
    refs2 = {}
    from ui.pages.settings.form_builder import build_form
    build_form(sp.current_ct, refs2)
    sp.current_ct.populate_form(refs2, cached)


def test_save_invalid_shows_errors():
    sp = _make_page()
    name_ctrl = _NameCtrl("")
    sp.config_name_ref.current = name_ctrl
    sp._switch_form("api")
    name_ctrl.value = "my_api"  # switch 已清空名称，补上以便走到校验阶段
    sp.field_refs["base_url"].value = ""
    sp._save(None)
    assert "校验失败" in sp.status_message
    assert sp.status_ok is False


def test_save_valid_writes_json(tmp_path):
    sp = _make_page()
    name_ctrl = _NameCtrl("")
    sp.config_name_ref.current = name_ctrl
    sp._switch_form("api")
    sp.config_name_ref.current = name_ctrl
    sp.field_refs["base_url"].value = "https://x.example"
    sp.field_refs["model"].value = "m"
    name_ctrl.value = "smoke_test"

    ct = sp.current_ct
    tmp = tmp_path
    orig_dir = ct.save_dir
    ct.save_dir = str(tmp)
    try:
        sp._save(None)
        assert sp.status_ok is True, sp.status_message
        files = list(tmp.glob("*.json"))
        assert len(files) == 1 and files[0].name == "smoke_test.json"
        data = json.loads(files[0].read_text(encoding="utf-8"))
        assert data["base_url"] == "https://x.example"
        assert data["model"] == "m"
        assert set(data) == {"base_url", "api_key", "model", "timeout"}
    finally:
        ct.save_dir = orig_dir


def test_safe_update_unmounted_no_raise():
    sp = _make_page()
    # 未挂载的控件 update 会抛 RuntimeError；_safe_update 应静默
    from flet import Text
    t = Text("x")
    sp._safe_update(t)  # 不应抛
    sp._safe_update(None)


def test_switch_from_ini_shows_file_list():
    """修复：初始系统条目（ini 托管）切出后，文件库中间列重建为文件列表。"""
    sp = _make_page()
    # 初始 translate_default（ini 托管）→ 文件库为 ini 卡片，无 file_list_col
    assert sp.current_ct.key == "translate_default"
    assert sp.file_list_col.current is None
    # 切出系统 → llama：文件列表出现且含该类型配置文件行
    sp._switch_form("llama")
    assert sp.file_list_col.current is not None
    rows = sp.file_list_col.current.controls
    assert rows, "llama 类型应有配置文件行"

    def row_names():
        out = []
        for r in rows:
            if hasattr(r, "controls") and r.controls:
                out.append(r.controls[0].content.value)
        return out

    names = row_names()
    assert any("default.json" in n for n in names), names


def test_switch_back_to_ini_shows_card():
    """json 类型切回系统条目 → 文件库恢复 ini 提示卡片（file_list_col 失效）。"""
    sp = _make_page()
    sp._switch_form("llama")
    assert sp.file_list_col.current is not None
    sp._switch_form("translate_default")
    assert sp.file_list_col.current is None


if __name__ == "__main__":
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
            passed += 1
        except Exception:
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(fns)} passed")
    sys.exit(0 if passed == len(fns) else 1)

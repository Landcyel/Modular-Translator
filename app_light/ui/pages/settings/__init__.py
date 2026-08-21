"""Settings page subpackage — config generator (two-column workbench layout).

Left column (2/5): config type navigation (top) + config file library (bottom)
Right column (3/5): config editor (grouped strongly-typed form)

After the declarative engine rewrite:
- Nav groups and all config types come from the single registry ``CONFIG_GROUPS`` (config_schema).
- Form rendering goes through ``form_builder.build_form``: group titles are rendered from
  ``ConfigType.groups``; errors are shown in place (no full-tree rebuild).
- Form refresh is unified into the single ``_render_form`` entry, shared by load/switch/new/validation-failure.
- The file library's import / export / duplicate buttons are wired (flet 0.86.2 async FilePicker).
- The browse button for path fields is injected via ``_browse_path`` (sync entry + page.run_task).
"""

import json
import flet as ft
from pathlib import Path

from ui.theme import Layout, Palette, Radius, Typography
from ui.components import _icon, _text, divider, _shadow, panel_header
from .config_schema import ALL_CONFIG_TYPES, CONFIG_TYPE_LIST, CONFIG_GROUPS, ConfigType
from .form_builder import build_form
from core.system_config import load_section, save_section


def build_settings(page: ft.Page, facade=None, file_picker: ft.FilePicker = None):
    """Compatibility wrapper — create a SettingsPage instance and build the UI."""
    return SettingsPage(page, facade, file_picker).build()


class SettingsPage:
    """Config generator page instance — holds state long-term so it survives navigation switches."""

    def __init__(self, page: ft.Page, facade=None, file_picker: ft.FilePicker = None):
        self.page = page
        self.facade = facade
        # flet 0.86.2: FilePicker is an async coroutine API (pick_files/get_directory_path/save_file)
        self.file_picker = file_picker or ft.FilePicker()
        # Register the FilePicker service (deferred to the page's first build/refresh, avoiding an update before page.add)
        self._picker_registered = False

        # ── Currently selected config type ──
        self.current_ct: ConfigType = CONFIG_TYPE_LIST[0]
        self.current_nav_key: str = self.current_ct.key

        # ── Form state ──
        self.field_refs: dict = {}
        self.current_form_values: dict | None = None
        self.config_name: str = ""
        self.status_message: str = ""
        self.status_ok: bool = True

        # ── Refs ──
        self.form_content = ft.Ref[ft.Column]()
        self.config_name_ref = ft.Ref[ft.TextField]()
        self.status_text = ft.Ref[ft.Text]()
        self.file_list_col = ft.Ref[ft.Column]()
        self.nav_column = ft.Ref[ft.Column]()
        self.editor_title = ft.Ref[ft.Text]()
        self.save_row_ref = ft.Ref[ft.Row]()
        # Strong reference to the top action row (flet 0.86.2 Ref is weak; a temporary Row
        # would be GC'd making current None; hold the mounted control via a normal attribute)
        self._save_row_ctrl: "ft.Row | None" = None
        # Strong reference to the file library panel (rebuilds the middle column when switching ini/json-managed types)
        self._file_panel_ctrl: "ft.Container | None" = None

        # ── Callback registration flag (settings page currently has no facade callbacks) ──
        self._callbacks_registered = False
        self.register_callbacks()

    # ── Public interface ──

    def build(self) -> ft.Control:
        """Build/rebuild the settings page UI (no top "Settings · Config Workbench" title bar)."""
        self._ensure_file_picker_registered()
        nav_panel = self._build_nav_panel()
        form_panel = self._build_editor_panel()
        file_panel = self._build_file_panel()
        return self._build_workspace(nav_panel, file_panel, form_panel)

    def save_ui_state(self) -> None:
        """Sync state from control refs to instance attributes before leaving the page."""
        if self.config_name_ref.current:
            self.config_name = self.config_name_ref.current.value or ""
        try:
            self.current_form_values = self.current_ct.form_values_from_refs(self.field_refs)
        except Exception:
            self.current_form_values = None

    def refresh(self) -> None:
        """Refresh on facade callbacks or when switching back to the page (restores the status-bar message)."""
        self._ensure_file_picker_registered()
        if self.status_message:
            self._set_status(self.status_message, self.status_ok)

    def _ensure_file_picker_registered(self) -> None:
        """Defer the FilePicker service registration (idempotent).

        In flet 0.86.2, service registration triggers page.update(); calling it before
        page.add(root) would patch an incomplete/empty control tree to the client and cause
        rendering issues. So registration is deferred to the page's first build/refresh
        (when the page is ready).
        """
        if self._picker_registered:
            return
        self._picker_registered = True
        if self.page is not None and hasattr(self.page, "_services"):
            try:
                self.page._services.register_service(self.file_picker)
            except Exception:
                pass  # swallow duplicate/not-ready registration

    def register_callbacks(self) -> None:
        """Register callbacks with the facade (the settings page currently has no facade callbacks)."""
        if self._callbacks_registered:
            return
        self._callbacks_registered = True

    # ── Internal methods — status bar ──

    def _safe_update(self, ctrl):
        """Defensive update: silently skip when the control is not mounted to a page (unmounted update raises RuntimeError)."""
        if ctrl is None:
            return
        try:
            if ctrl.page:
                ctrl.update()
        except RuntimeError:
            pass

    def _set_status(self, msg: str, ok: bool = True):
        self.status_message = msg
        self.status_ok = ok
        if self.status_text.current:
            self.status_text.current.value = msg
            self.status_text.current.color = Palette.SUCCESS if ok else Palette.ERROR
            self._safe_update(self.status_text.current)

    # ── Internal methods — panel building ──

    def _build_nav_panel(self) -> ft.Container:
        nav_items = self._build_nav_items()
        return ft.Container(
            content=ft.Column(
                controls=[
                    panel_header("配置类型"),
                    divider(),
                    ft.Column(
                        ref=self.nav_column,
                        controls=nav_items,
                        spacing=2,
                        scroll=ft.ScrollMode.AUTO,
                        expand=True,
                    ),
                ],
                spacing=8,
                expand=True,
            ),
            bgcolor=Palette.SURFACE,
            border_radius=Radius.XL,
            padding=ft.Padding.all(14),
            border=ft.Border.all(1, Palette.BORDER_SUBTLE),
            shadow=_shadow("low"),
            expand=3,
        )

    def _build_save_row(self) -> ft.Row:
        """Top action row: json-managed types = config name + New + Save;
        ini-managed types (default configs) = explanation text + Save (writes default.ini directly)."""
        save_btn = ft.FilledButton(
            content=ft.Text("保存配置", size=12, color="#FFFFFF"),
            on_click=self._save,
            style=ft.ButtonStyle(
                bgcolor=Palette.PRIMARY, color="#FFFFFF",
                shape=ft.RoundedRectangleBorder(radius=10),
                padding=ft.Padding.symmetric(horizontal=16, vertical=6),
            ),
        )
        if self.current_ct.ini_section:
            return ft.Row([
                ft.Text("默认配置统一由 configs/system/default.ini 管理，保存后立即生效",
                        size=12, color=Palette.SUBTEXT),
                ft.Container(expand=True),
                save_btn,
            ], ref=self.save_row_ref, spacing=12,
               vertical_alignment=ft.CrossAxisAlignment.CENTER)
        return ft.Row([
            ft.Text("配置名称", size=12, weight=ft.FontWeight.W_500,
                    color=Palette.SUBTEXT, width=80),
            ft.TextField(
                ref=self.config_name_ref, value=self.config_name, dense=True,
                hint_text="输入名称，如 my_config",
                hint_style=ft.TextStyle(size=12, color=Palette.SUBTEXT),
                text_style=ft.TextStyle(size=13, color=Palette.TEXT),
                border_color=Palette.BORDER, border_radius=8,
                bgcolor=Palette.BG, width=240,
            ),
            ft.Container(expand=True),
            ft.OutlinedButton(
                content=ft.Text("新建", size=12, color=Palette.PRIMARY),
                on_click=self._new_config,
                style=ft.ButtonStyle(
                    color=Palette.PRIMARY,
                    shape=ft.RoundedRectangleBorder(radius=10),
                    side=ft.BorderSide(1, Palette.PRIMARY),
                    padding=ft.Padding.symmetric(horizontal=12, vertical=6),
                ),
            ),
            save_btn,
        ], ref=self.save_row_ref, spacing=12,
           vertical_alignment=ft.CrossAxisAlignment.CENTER)

    def _refresh_save_row(self) -> None:
        """Rebuild the top action row when switching ini-managed types.

        Note: the temporary Row used for refresh no longer binds a Ref (flet 0.86.2 Ref is weak;
        after the temporary control is GC'd, current becomes None); update the mounted control
        held by the strong reference directly.
        """
        if self._save_row_ctrl is not None:
            self._save_row_ctrl.controls = self._build_save_row().controls
            self._safe_update(self._save_row_ctrl)

    def _build_editor_panel(self) -> ft.Container:
        save_row = self._build_save_row()
        # Hold a reference to the mounted control (instead of Ref, avoiding weak-ref invalidation)
        self._save_row_ctrl = save_row

        # Initialize the form (on first build form_content ref is not mounted; construct directly)
        self.field_refs = {}
        initial_rows = build_form(
            self.current_ct, self.field_refs,
            values=self.current_form_values,
            on_browse=self._browse_path,
        )

        return ft.Container(
            content=ft.Column([
                # ── Top fixed: title + save status ──
                ft.Row([
                    ft.Text(ref=self.editor_title,
                            value=self._build_editor_title(),
                            size=Typography.HEADING_SM, weight=ft.FontWeight.BOLD,
                            color=Palette.TEXT),
                    ft.Container(expand=True),
                    ft.Text(ref=self.status_text, size=Typography.BODY_SM,
                            color=Palette.SUCCESS),
                ], spacing=10),
                # ── Top fixed: config name + save actions ──
                save_row,
                divider(),
                # ── Middle scrollable: grouped form ──
                ft.Column(ref=self.form_content,
                          controls=initial_rows,
                          spacing=14,
                          expand=True,
                          scroll=ft.ScrollMode.AUTO),
            ], spacing=12,
               expand=True),
            bgcolor=Palette.SURFACE,
            border_radius=Radius.XL,
            padding=ft.Padding.all(Layout.CARD_PADDING),
            border=ft.Border.all(1, Palette.BORDER_SUBTLE),
            shadow=_shadow("low"),
            expand=3,
        )

    def _build_ini_card(self) -> ft.Container:
        """Info card for the file library of ini-managed types (default configs)."""
        return ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Text("默认配置由 INI 统一管理", size=13,
                            weight=ft.FontWeight.BOLD, color=Palette.TEXT),
                ], spacing=8),
                ft.Text(
                    f"当前条目（{self.current_ct.label}）的默认设置保存在 "
                    f"configs/system/default.ini 的 [{self.current_ct.ini_section}] 段。\n\n"
                    "在右侧表单修改后点击「保存配置」即直接写入该文件并立即生效；"
                    "也可手动编辑 INI 文件，重新打开设置页后生效。",
                    size=12, color=Palette.SUBTEXT,
                ),
            ], spacing=8),
            bgcolor=f"{Palette.PRIMARY}0D",
            border=ft.Border.all(1, Palette.PRIMARY + "33"),
            border_radius=10,
            padding=ft.Padding.all(12),
        )

    def _build_file_panel_inner(self) -> ft.Column:
        """File library middle column: ini-managed types show an info card; json types show the file list + action row."""
        if self.current_ct.ini_section:
            return ft.Column([
                panel_header("配置文件库"),
                divider(),
                self._build_ini_card(),
            ], spacing=8, expand=True)
        initial_file_rows = self._build_file_rows()
        return ft.Column([
            panel_header("配置文件库"),
            divider(),
            ft.Column(ref=self.file_list_col, spacing=4,
                      controls=initial_file_rows,
                      expand=True,
                      scroll=ft.ScrollMode.AUTO),
            divider(),
            ft.Row([
                ft.TextButton(
                    "导入配置",
                    on_click=self._import_config,
                    style=ft.ButtonStyle(color=Palette.SUBTEXT, padding=ft.Padding.all(4)),
                ),
                ft.TextButton(
                    "导出配置",
                    on_click=self._export_config,
                    style=ft.ButtonStyle(color=Palette.SUBTEXT, padding=ft.Padding.all(4)),
                ),
                ft.Container(expand=True),
                ft.TextButton(
                    "复制当前配置",
                    on_click=self._duplicate_config,
                    style=ft.ButtonStyle(color=Palette.SUBTEXT, padding=ft.Padding.all(4)),
                ),
            ], spacing=2),
        ], spacing=8,
           expand=True)

    def _build_file_panel(self) -> ft.Container:
        container = ft.Container(
            content=self._build_file_panel_inner(),
            bgcolor=Palette.SURFACE,
            border_radius=Radius.XL,
            padding=ft.Padding.all(14),
            border=ft.Border.all(1, Palette.BORDER_SUBTLE),
            shadow=_shadow("low"),
            expand=2,
        )
        # Hold the mounted control (Ref weak refs go stale), for _refresh_file_panel to rebuild the middle column
        self._file_panel_ctrl = container
        return container

    def _refresh_file_panel(self) -> None:
        """Rebuild the file library middle column when switching ini/json-managed types (ini card ↔ file list).

        After the rebuild, the file_list_col ref rebinds to the new list Column (the ini branch
        has no file_list_col → current is None, so _refresh_file_list safely returns).
        """
        if self._file_panel_ctrl is None:
            return
        self._file_panel_ctrl.content = self._build_file_panel_inner()
        self._safe_update(self._file_panel_ctrl)

    def _build_workspace(self, nav_panel, file_panel, form_panel) -> ft.Control:
        """Left column (nav + file library) + right column (editor); stacked vertically on narrow screens."""
        left_col = ft.Column([
            nav_panel,
            ft.Container(height=Layout.COLUMN_SPACING),
            file_panel,
        ], expand=True, spacing=0)

        is_narrow = self.page.width > 0 and self.page.width < Layout.DESKTOP_MIN_WIDTH
        if is_narrow:
            workspace = ft.Column([
                left_col,
                ft.Container(height=Layout.COLUMN_SPACING),
                form_panel,
            ], expand=True, spacing=0, scroll=ft.ScrollMode.AUTO)
        else:
            workspace = ft.Row([
                left_col,
                ft.Container(width=Layout.COLUMN_SPACING),
                form_panel,
            ], expand=True,
               vertical_alignment=ft.CrossAxisAlignment.STRETCH)
        left_col.expand = 2
        form_panel.expand = 3
        return workspace

    # ── Internal methods — unified form refresh ──

    def _typed_to_form_values(self, typed: dict) -> dict:
        """Typed intermediate dict → control-value dict (used to restore the form after validation failure)."""
        return {f.key: f.to_control_value(typed.get(f.key, f.default))
                for f in self.current_ct.fields}

    def _render_form(self, form_values: dict | None = None,
                     errors: dict | None = None) -> None:
        """Rebuild the form area and restore state — single entry shared by load/switch/new/validation-failure."""
        self.field_refs = {}
        rows = build_form(self.current_ct, self.field_refs,
                          values=form_values, errors=errors,
                          on_browse=self._browse_path,
                          on_change=self._on_form_select_change)
        if self.form_content.current:
            self.form_content.current.controls = rows
            self._safe_update(self.form_content.current)
        if self.editor_title.current:
            self.editor_title.current.value = self._build_editor_title()
            self._safe_update(self.editor_title.current)

    def _on_form_select_change(self, e):
        """select change linkage: the mode field → updates current form values and re-renders (visible_when show/hide)."""
        key = getattr(e.control, "data", None)
        if key != "mode":
            return
        values = dict(self.current_form_values or {})
        values["mode"] = getattr(e.control, "value", "default")
        self.current_form_values = values
        self._render_form(form_values=values)

    def _browse_path(self, field, ref) -> None:
        """Browse button for path fields — sync entry; dispatches the async FilePicker internally (flet 0.86.2)."""
        if self.file_picker is None or self.page is None:
            return

        async def _do():
            try:
                if field.browse == "directory":
                    path = await self.file_picker.get_directory_path(
                        dialog_title=f"选择 {field.label}")
                else:
                    # flet 0.86.2: pick_files returns list[FilePickerFile] directly
                    files = await self.file_picker.pick_files(
                        dialog_title=f"选择 {field.label}",
                        file_type=ft.FilePickerFileType.ANY)
                    path = files[0].path if files else None
                if path:
                    ref.value = path
                    ref.update()
            except Exception as ex:
                self._set_status(f"✗ 选择失败: {ex}", ok=False)

        self.page.run_task(_do)

    # ── Internal methods — navigation rail ──

    def _build_editor_title(self) -> str:
        if self.config_name:
            return f"{self.current_ct.label} · {self.config_name}"
        return self.current_ct.label

    def _build_nav_items(self) -> list:
        items = []
        for group_name, type_items in CONFIG_GROUPS:
            # Group header item: larger hollow dot + font size bumped up one step + lighter color
            items.append(
                ft.Container(
                    content=ft.Row([
                        _icon(ft.Icons.CIRCLE_OUTLINED, size=12,
                              color=Palette.TEXT_SECOND),
                        ft.Text(group_name, size=12, weight=ft.FontWeight.W_700,
                                color=Palette.TEXT_SECOND),
                    ], spacing=6),
                    padding=ft.Padding.only(left=10, top=16, bottom=4),
                )
            )
            for ct in type_items:
                is_active = (ct.key == self.current_nav_key)
                items.append(
                    ft.Container(
                        content=ft.Row([
                            ft.Text(ct.label, size=13,
                                    weight=(ft.FontWeight.W_600 if is_active
                                            else ft.FontWeight.NORMAL),
                                    color=Palette.TEXT if is_active else Palette.SUBTEXT),
                        ], spacing=8),
                        bgcolor=Palette.PRIMARY + "18" if is_active else None,
                        border_radius=8,
                        padding=ft.Padding.symmetric(horizontal=10, vertical=8),
                        ink=True,
                        on_click=lambda e, k=ct.key: self._switch_form(k),
                    )
                )
        return items

    def _refresh_nav(self):
        if self.nav_column.current:
            self.nav_column.current.controls = self._build_nav_items()
            self._safe_update(self.nav_column.current)

    def _switch_form(self, ctype_key: str):
        target = ALL_CONFIG_TYPES.get(ctype_key)
        if target is None:
            return
        self.current_ct = target
        self.current_nav_key = self.current_ct.key
        self.config_name = ""
        # ini-managed type (default config): load the corresponding ini section to fill the form
        if target.ini_section:
            self.current_form_values = target.to_form_values(
                load_section(target.ini_section))
        else:
            self.current_form_values = None
        self._render_form(form_values=self.current_form_values)
        if self.config_name_ref.current:
            self.config_name_ref.current.value = ""
            self._safe_update(self.config_name_ref.current)
        self._set_status("")
        self._refresh_save_row()
        self._refresh_file_panel()
        self._refresh_nav()

    def _new_config(self, e):
        self.current_form_values = None
        self.config_name = ""
        self._render_form(form_values=None)
        if self.config_name_ref.current:
            self.config_name_ref.current.value = ""
            self._safe_update(self.config_name_ref.current)
        self._set_status("")

    # ── Internal methods — file list ──

    def _list_config_files(self, ct: ConfigType) -> list[Path]:
        save_dir = ct.save_path
        if not save_dir.exists():
            return []
        return sorted(save_dir.glob(getattr(ct, "name_filter", "*.json")),
                      key=lambda p: p.name)

    def _build_file_rows(self) -> list:
        files = self._list_config_files(self.current_ct)
        rows = []
        if not files:
            rows.append(
                ft.Text("  (暂无配置文件)", size=12, italic=True,
                        color=Palette.SUBTEXT)
            )
        else:
            for fp in files:
                fname = fp.name
                rows.append(
                    ft.Row([
                        ft.TextButton(
                            content=ft.Text(fname, size=13, color=Palette.PRIMARY),
                            style=ft.ButtonStyle(
                                color=Palette.PRIMARY, padding=ft.Padding.all(0),
                            ),
                            on_click=lambda e, fn=fname: self._load_config(fn),
                        ),
                        ft.Container(expand=True),
                        ft.TextButton(
                            "删除",
                            on_click=lambda e, fn=fname: self._confirm_delete(fn),
                            style=ft.ButtonStyle(
                                color=Palette.ERROR, padding=ft.Padding.symmetric(horizontal=8),
                            ),
                        ),
                    ], spacing=4, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                )
        return rows

    def _refresh_file_list(self):
        if self.file_list_col.current is None:
            return
        self.file_list_col.current.controls = self._build_file_rows()
        self._safe_update(self.file_list_col.current)

    # ── Internal methods — config load/import/export/duplicate/delete ──

    def _load_config(self, filename: str):
        fp = self.current_ct.save_path / filename
        if not fp.exists():
            self._set_status(f"✗ 文件不存在: {filename}", ok=False)
            return
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as ex:
            self._set_status(f"✗ 读取失败: {ex}", ok=False)
            return

        form_values = self.current_ct.to_form_values(data)
        self._render_form(form_values=form_values)
        if self.config_name_ref.current:
            self.config_name = fp.stem
            self.config_name_ref.current.value = fp.stem
            self._safe_update(self.config_name_ref.current)
        self._set_status(f"✓ 已加载 {filename}", ok=True)

    def _import_config(self, e):
        """Import an external JSON into the current type's form (not written to disk; user must save)."""
        if self.file_picker is None or self.page is None:
            return

        async def _do():
            try:
                # flet 0.86.2: pick_files returns list[FilePickerFile] directly
                files = await self.file_picker.pick_files(
                    dialog_title="导入配置文件",
                    file_type=ft.FilePickerFileType.ANY,
                    allowed_extensions=["json"])
                if not files:
                    return
                path = Path(files[0].path)
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as ex:
                self._set_status(f"✗ 导入失败: {ex}", ok=False)
                return
            form_values = self.current_ct.to_form_values(data)
            self._render_form(form_values=form_values)
            if self.config_name_ref.current:
                self.config_name = path.stem
                self.config_name_ref.current.value = path.stem
                self._safe_update(self.config_name_ref.current)
            self._set_status(f"✓ 已导入 {path.name}，保存后写入 {self.current_ct.save_dir}", ok=True)

        self.page.run_task(_do)

    def _export_config(self, e):
        """Export the current form build result to a JSON file."""
        if self.file_picker is None or self.page is None:
            return
        name = ((self.config_name_ref.current.value or "").strip()
                if self.config_name_ref.current else "")
        base = name or self.current_ct.key

        async def _do():
            try:
                path = await self.file_picker.save_file(
                    dialog_title="导出配置",
                    file_name=f"{base}.json",
                    file_type=ft.FilePickerFileType.ANY,
                    allowed_extensions=["json"])
                if not path:
                    return
                intermediate, parse_errors = self.current_ct.collect_values(self.field_refs)
                field_errors, general = self.current_ct.validate(intermediate)
                if parse_errors or field_errors or general:
                    self._set_status("✗ 表单存在错误，无法导出", ok=False)
                    return
                result = self.current_ct.build_output(intermediate)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False, indent=2))
                self._set_status(f"✓ 已导出至 {path}", ok=True)
            except Exception as ex:
                self._set_status(f"✗ 导出失败: {ex}", ok=False)

        self.page.run_task(_do)

    def _duplicate_config(self, e):
        """Duplicate the current form values as a new config: name box set to {original}_copy, not written to disk."""
        name = ((self.config_name_ref.current.value or "").strip()
                if self.config_name_ref.current else "")
        new_name = f"{name}_copy" if name else f"{self.current_ct.key}_copy"
        self.config_name = new_name
        if self.config_name_ref.current:
            self.config_name_ref.current.value = new_name
            self._safe_update(self.config_name_ref.current)
        self._set_status(f"✓ 已复制当前配置，保存为 {new_name}", ok=True)

    def _confirm_delete(self, filename: str):
        fp = self.current_ct.save_path / filename

        def _close_dlg():
            dlg.open = False
            self.page.update()

        def _do_delete(e):
            try:
                fp.unlink()
                self._set_status(f"✓ 已删除 {filename}", ok=True)
                self._refresh_file_list()
            except Exception as ex:
                self._set_status(f"✗ 删除失败: {ex}", ok=False)
            _close_dlg()

        dlg = ft.AlertDialog(
            modal=True,
            title=ft.Text("确认删除"),
            content=ft.Text(f"确定要删除配置文件 \"{filename}\" 吗？此操作不可恢复。"),
            actions=[
                ft.TextButton(content=ft.Text("取消"), on_click=lambda e: _close_dlg()),
                ft.FilledButton(content=ft.Text("删除", color="#FFFFFF"), on_click=_do_delete,
                                style=ft.ButtonStyle(bgcolor=Palette.ERROR, color="#FFFFFF")),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )
        self.page.show_dialog(dlg)

    # ── Internal methods — save ──

    def _confirm_overwrite(self, fp: Path, json_str: str):
        def _close_dlg():
            dlg.open = False
            self.page.update()

        def _do_write(e):
            self._write_file(fp, json_str)
            _close_dlg()

        dlg = ft.AlertDialog(
            modal=True,
            title=ft.Text("文件已存在"),
            content=ft.Text(f"配置文件 \"{fp.name}\" 已存在，是否覆盖？"),
            actions=[
                ft.TextButton(content=ft.Text("取消"), on_click=lambda e: _close_dlg()),
                ft.FilledButton(content=ft.Text("覆盖", color="#FFFFFF"), on_click=_do_write,
                                style=ft.ButtonStyle(bgcolor=Palette.PRIMARY, color="#FFFFFF")),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )
        self.page.show_dialog(dlg)

    def _write_file(self, fp: Path, json_str: str):
        try:
            fp.parent.mkdir(parents=True, exist_ok=True)
            with open(fp, "w", encoding="utf-8") as f:
                f.write(json_str)
            rel_path = fp.relative_to(Path.cwd()) if fp.is_relative_to(Path.cwd()) else fp
            self._set_status(f"✓ 配置已保存至 {rel_path}", ok=True)
            self._refresh_file_list()
            self.config_name = fp.stem
            if self.editor_title.current:
                self.editor_title.current.value = self._build_editor_title()
                self._safe_update(self.editor_title.current)
        except Exception as ex:
            self._set_status(f"✗ 写入失败: {ex}", ok=False)

    def _save(self, e):
        ct = self.current_ct
        name = ((self.config_name_ref.current.value or "").strip()
                if self.config_name_ref.current else "")

        try:
            intermediate, parse_errors = ct.collect_values(self.field_refs)
            field_errors, general_errors = ct.validate(intermediate)
            if parse_errors or field_errors or general_errors:
                errors = {**parse_errors, **field_errors}
                if general_errors:
                    errors["_general"] = general_errors
                self._render_form(form_values=self._typed_to_form_values(intermediate),
                                  errors=errors)
                n = len(parse_errors) + len(field_errors) + len(general_errors)
                self._set_status(f"✗ 校验失败: {n} 项错误", ok=False)
                return

            result = ct.build_output(intermediate)

            # ini-managed type (default config): write directly to the configs/system/default.ini section
            if ct.ini_section:
                save_section(ct.ini_section, result)
                self._set_status(
                    f"✓ 已保存至 configs/system/default.ini [{ct.ini_section}]", ok=True)
                return

            if not name:
                self._set_status("✗ 请先输入配置名称", ok=False)
                return

            json_str = json.dumps(result, ensure_ascii=False, indent=2)
            fp = ct.save_path / f"{name}.json"

            if fp.exists():
                self._confirm_overwrite(fp, json_str)
            else:
                self._write_file(fp, json_str)
        except Exception as ex:
            self._set_status(f"✗ 保存失败: {ex}", ok=False)

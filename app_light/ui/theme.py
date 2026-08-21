"""配色常量与设计令牌 — 仅深色主题。"""


class Palette:
    """深色色板。"""
    PRIMARY        = "#6366F1"
    PRIMARY_LIGHT  = "#818CF8"
    PRIMARY_DARK   = "#4F46E5"
    PRIMARY_GLOW   = "#6366F1"

    SUCCESS = "#10B981"
    WARNING = "#F59E0B"
    ERROR   = "#EF4444"
    INFO    = "#3B82F6"

    BG            = "#0B1120"
    SURFACE       = "#141C30"
    SURFACE2      = "#1C2640"
    SURFACE3      = "#26344F"
    SURFACE_HOVER = "#1A253C"

    TEXT       = "#F1F5F9"
    TEXT_SECOND = "#CBD5E1"
    SUBTEXT    = "#94A3B8"
    TEXT_MUTED = "#64748B"

    BORDER        = "#2A3A55"
    BORDER_SUBTLE = "#1E2A40"

    GRADIENT_1 = ["#6366F1", "#8B5CF6"]
    GRADIENT_2 = ["#10B981", "#34D399"]
    GRADIENT_3 = ["#F59E0B", "#FBBF24"]
    GRADIENT_4 = ["#EF4444", "#F87171"]


class Typography:
    """字号比例令牌。"""
    CAPTION    = 10
    SMALL      = 11
    BODY_SM    = 12
    BODY       = 13
    BODY_LG    = 14
    HEADING_SM = 15
    HEADING    = 16
    HEADING_LG = 17
    TITLE      = 20
    DISPLAY    = 28


class Radius:
    """圆角令牌。"""
    XS  = 6
    SM  = 8
    MD  = 10
    LG  = 12
    XL  = 16


class Anim:
    """动效令牌。"""
    FAST    = 50
    NORMAL  = 50
    SLOW    = 50
    CURVE   = "ease"


class Layout:
    """可复用布局尺寸令牌 — 统一各页面间距/内边距/高度等数值。"""
    # 内容区域内统一间距：复合组件之间、组件到内容区域边界均使用此值
    # （= 翻译页「源文本」面板与「服务栏」之间的间距，全局一致）
    CONTENT_GAP       = 16
    PAGE_PADDING      = CONTENT_GAP   # 内容区到窗口边界的内边距
    SECTION_GAP       = CONTENT_GAP   # 段（复合组件）间距
    COLUMN_SPACING    = CONTENT_GAP   # 双栏/纵向列间距

    CARD_PADDING      = 20
    CARD_RADIUS       = 16
    MINI_CARD_PADDING = 14
    MINI_CARD_RADIUS  = 12

    CONTROL_HEIGHT    = 36
    ICON_BTN_SIZE     = 32

    WORKSPACE_HEIGHT           = 520
    SETTINGS_WORKSPACE_HEIGHT  = 600

    BRAND_HEIGHT   = 68   # 品牌区固定高度
    APP_BAR_HEIGHT = 34   # 顶栏固定高度 = 品牌区一半
    # 侧栏（品牌区/导航栏/左列）统一宽度：220 时品牌区文本列仅 142px（220-32 padding-36 图标-10 spacing），
    # 放不下 16px 粗体 "Modular Translator"（约需 150px+）；加宽至 250 后可用约 172px，保证文字完整显示
    SIDEBAR_WIDTH  = 250

    COLUMN_SPACING    = 16

    DESKTOP_MIN_WIDTH = 960

    PICKER_WIDTH_LG   = 240
    PICKER_WIDTH_SM   = 190
    PICKER_DD_MINUS   = 40


def palette_colors() -> dict:
    return {
        "bg":            Palette.BG,
        "surface":       Palette.SURFACE,
        "surface2":      Palette.SURFACE2,
        "surface3":      Palette.SURFACE3,
        "surface_hover": Palette.SURFACE_HOVER,
        "text":          Palette.TEXT,
        "text_second":   Palette.TEXT_SECOND,
        "subtext":       Palette.SUBTEXT,
        "text_muted":    Palette.TEXT_MUTED,
        "border":        Palette.BORDER,
        "border_subtle": Palette.BORDER_SUBTLE,
    }

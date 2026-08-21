"""Color constants and design tokens — dark theme only."""


class Palette:
    """Dark palette."""
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
    """Font size ratio tokens."""
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
    """Corner radius tokens."""
    XS  = 6
    SM  = 8
    MD  = 10
    LG  = 12
    XL  = 16


class Anim:
    """Animation tokens."""
    FAST    = 50
    NORMAL  = 50
    SLOW    = 50
    CURVE   = "ease"


class Layout:
    """Reusable layout size tokens — unify spacing/padding/heights across pages."""
    # Uniform spacing inside the content area: used between composite components and
    # from components to the content-area boundary (= the gap between the Source Text
    # panel and the service bar on the Translate page, applied globally)
    CONTENT_GAP       = 16
    PAGE_PADDING      = CONTENT_GAP   # padding from the content area to the window edge
    SECTION_GAP       = CONTENT_GAP   # section (composite component) spacing
    COLUMN_SPACING    = CONTENT_GAP   # two-column / vertical column spacing

    CARD_PADDING      = 20
    CARD_RADIUS       = 16
    MINI_CARD_PADDING = 14
    MINI_CARD_RADIUS  = 12

    CONTROL_HEIGHT    = 36
    ICON_BTN_SIZE     = 32

    WORKSPACE_HEIGHT           = 520
    SETTINGS_WORKSPACE_HEIGHT  = 600

    BRAND_HEIGHT   = 68   # Fixed brand area height
    APP_BAR_HEIGHT = 34   # Fixed top bar height = half the brand area
    # Unified sidebar width (brand/nav/left column): at 220 the brand text column is
    # only 142px (220-32 padding-36 icon-10 spacing), too narrow for 16px bold
    # "Modular Translator" (~150px+); widened to 250 gives ~172px so the text fits fully
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

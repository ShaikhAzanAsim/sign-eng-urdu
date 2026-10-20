"""Urdu/Arabic compositor. OpenCV putText cannot shape RTL joined letterforms.

Modern Noto Nastaliq stores outlines behind OpenType GSUB, so Pillow+arabic_reshaper
draws tofu. Shape with HarfBuzz, rasterize glyph ids with FreeType, then alpha-paste.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import numpy as np

from src.console import configure_utf8_stdout
from src.paths import FONTS_DIR

NOTO_NASTALIQ_URLS = (
    "https://raw.githubusercontent.com/google/fonts/main/ofl/notonastaliqurdu/NotoNastaliqUrdu%5Bwght%5D.ttf",
)
FONT_PATH = FONTS_DIR / "NotoNastaliqUrdu-Regular.ttf"
_DOWNLOADED_VARIABLE = FONTS_DIR / "NotoNastaliqUrdu-variable.ttf"

_hb_face = None
_ft_face = None
_font_bytes: bytes | None = None


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "sign-urdu-eng/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp, tmp.open("wb") as fh:
        fh.write(resp.read())
    tmp.replace(dest)
    return dest


def _instantiate_regular(variable_path: Path, dest: Path) -> Path:
    from fontTools.ttLib import TTFont
    from fontTools.varLib.instancer import instantiateVariableFont

    font = TTFont(str(variable_path))
    if "fvar" not in font:
        dest.write_bytes(variable_path.read_bytes())
        return dest
    static = instantiateVariableFont(font, {"wght": 400})
    dest.parent.mkdir(parents=True, exist_ok=True)
    static.save(str(dest))
    return dest


def ensure_urdu_font(dest: Path | None = None) -> Path:
    path = dest or FONT_PATH
    if path.exists() and path.stat().st_size > 10_000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for url in NOTO_NASTALIQ_URLS:
        print(f"Downloading Noto Nastaliq Urdu from {url} ...")
        try:
            _download(url, _DOWNLOADED_VARIABLE)
            return _instantiate_regular(_DOWNLOADED_VARIABLE, path)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not download Noto Nastaliq Urdu: {last_error}")


def _engines():
    global _hb_face, _ft_face, _font_bytes
    import freetype
    import uharfbuzz as hb

    path = ensure_urdu_font()
    if _font_bytes is None:
        _font_bytes = path.read_bytes()
        _hb_face = hb.Face(_font_bytes)
        _ft_face = freetype.Face(str(path))
    return _hb_face, _ft_face


def contains_arabic(text: str) -> bool:
    return any("\u0600" <= ch <= "\u06FF" for ch in text)


def _shaped_rgba(
    text: str,
    font_size: int,
    fill_bgr: tuple[int, int, int],
    stroke_bgr: tuple[int, int, int],
    stroke_width: int,
) -> np.ndarray | None:
    import freetype
    import uharfbuzz as hb

    hb_face, ft_face = _engines()
    font = hb.Font(hb_face)
    font.scale = (font_size, font_size)
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(font, buf)

    ft_face.set_pixel_sizes(0, font_size)
    pen_x = 0
    pen_y = 0
    stamps: list[tuple[int, int, np.ndarray]] = []
    minx, miny = 10**9, 10**9
    maxx, maxy = -10**9, -10**9

    for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
        ft_face.load_glyph(info.codepoint, freetype.FT_LOAD_RENDER)
        bitmap = ft_face.glyph.bitmap
        if bitmap.width == 0 or bitmap.rows == 0:
            pen_x += int(pos.x_advance)
            pen_y += int(pos.y_advance)
            continue
        arr = np.frombuffer(bytes(bitmap.buffer), dtype=np.uint8)
        arr = arr.reshape(bitmap.rows, bitmap.pitch)[:, : bitmap.width]
        x = int(pen_x + pos.x_offset + ft_face.glyph.bitmap_left)
        y = int(pen_y + pos.y_offset - ft_face.glyph.bitmap_top)
        stamps.append((x, y, arr))
        minx = min(minx, x)
        miny = min(miny, y)
        maxx = max(maxx, x + arr.shape[1])
        maxy = max(maxy, y + arr.shape[0])
        pen_x += int(pos.x_advance)
        pen_y += int(pos.y_advance)

    if not stamps:
        return None

    pad = max(stroke_width, 2)
    width = maxx - minx + pad * 2
    height = maxy - miny + pad * 2
    alpha = np.zeros((height, width), dtype=np.uint8)
    for x, y, arr in stamps:
        x0 = x - minx + pad
        y0 = y - miny + pad
        roi = alpha[y0 : y0 + arr.shape[0], x0 : x0 + arr.shape[1]]
        np.maximum(roi, arr, out=roi)

    if stroke_width > 0:
        kernel = np.ones((stroke_width * 2 + 1, stroke_width * 2 + 1), dtype=np.uint8)
        outline = cv2.dilate(alpha, kernel)
        overlay = np.zeros((height, width, 4), dtype=np.uint8)
        overlay[:, :, 0] = stroke_bgr[0]
        overlay[:, :, 1] = stroke_bgr[1]
        overlay[:, :, 2] = stroke_bgr[2]
        overlay[:, :, 3] = outline
        ink = np.zeros((height, width, 4), dtype=np.uint8)
        ink[:, :, 0] = fill_bgr[0]
        ink[:, :, 1] = fill_bgr[1]
        ink[:, :, 2] = fill_bgr[2]
        ink[:, :, 3] = alpha
        overlay = _over(overlay, ink)
        return overlay

    overlay = np.zeros((height, width, 4), dtype=np.uint8)
    overlay[:, :, 0] = fill_bgr[0]
    overlay[:, :, 1] = fill_bgr[1]
    overlay[:, :, 2] = fill_bgr[2]
    overlay[:, :, 3] = alpha
    return overlay


def _over(base: np.ndarray, top: np.ndarray) -> np.ndarray:
    a = top[:, :, 3:4].astype(np.float32) / 255.0
    out = base.astype(np.float32)
    out[:, :, :3] = a * top[:, :, :3] + (1.0 - a) * out[:, :, :3]
    out[:, :, 3:4] = np.clip(top[:, :, 3:4].astype(np.float32) + (1.0 - a) * out[:, :, 3:4], 0, 255)
    return out.astype(np.uint8)


def render_urdu_text(
    frame: np.ndarray,
    text: str,
    position: tuple[int, int],
    font_size: int = 36,
    color: tuple[int, int, int] = (255, 255, 255),
    stroke_color: tuple[int, int, int] = (0, 0, 0),
    stroke_width: int = 2,
    anchor: str = "lt",
) -> np.ndarray:
    """Composite shaped Urdu onto a BGR OpenCV frame. Returns the same array."""
    if not text:
        return frame

    overlay = _shaped_rgba(text, font_size, color, stroke_color, stroke_width)
    if overlay is None:
        return frame

    h, w = overlay.shape[:2]
    x, y = position
    if "r" in anchor:
        x -= w
    elif "m" in anchor or anchor in ("mt", "mb", "mm"):
        x -= w // 2
    if "b" in anchor:
        y -= h
    elif anchor in ("lm", "rm", "mm"):
        y -= h // 2

    _alpha_paste(frame, overlay, x, y)
    return frame


def _alpha_paste(frame_bgr: np.ndarray, overlay_bgra: np.ndarray, x: int, y: int) -> None:
    fh, fw = frame_bgr.shape[:2]
    oh, ow = overlay_bgra.shape[:2]
    x0 = max(x, 0)
    y0 = max(y, 0)
    x1 = min(x + ow, fw)
    y1 = min(y + oh, fh)
    if x1 <= x0 or y1 <= y0:
        return
    ox0 = x0 - x
    oy0 = y0 - y
    slice_bg = frame_bgr[y0:y1, x0:x1]
    slice_fg = overlay_bgra[oy0 : oy0 + (y1 - y0), ox0 : ox0 + (x1 - x0)]
    alpha = slice_fg[:, :, 3:4].astype(np.float32) / 255.0
    bgr = slice_fg[:, :, :3].astype(np.float32)
    out = alpha * bgr + (1.0 - alpha) * slice_bg.astype(np.float32)
    slice_bg[:] = out.astype(np.uint8)


def main() -> None:
    configure_utf8_stdout()
    ensure_urdu_font()
    frame = np.full((240, 780, 3), 30, dtype=np.uint8)
    render_urdu_text(frame, "اردو حروف تہجی", (40, 70), font_size=48)
    out = FONTS_DIR.parent / "data" / "references" / "urdu_render_test.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), frame)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

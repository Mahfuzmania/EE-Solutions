from __future__ import annotations

from io import BytesIO
from typing import Dict, List

from PIL import Image, ImageDraw

try:
    import schemdraw
    import schemdraw.elements as elm
except Exception:  # pragma: no cover
    schemdraw = None
    elm = None


def _placeholder(text: str) -> bytes:
    img = Image.new("RGB", (800, 220), color=(25, 35, 48))
    d = ImageDraw.Draw(img)
    d.text((20, 20), text, fill=(240, 240, 240))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def render_series(components: List[str], labels: List[str] | None = None) -> bytes:
    if schemdraw is None:
        return _placeholder("Schemdraw not installed. Cannot render circuit.")

    if not components:
        return _placeholder("No components provided.")

    labels = labels or ["" for _ in components]
    with schemdraw.Drawing() as d:
        d += elm.SourceV().label("V")
        for comp, label in zip(components, labels):
            comp = comp.upper().strip()
            if comp == "R":
                d += elm.Resistor().label(label or "R")
            elif comp == "L":
                d += elm.Inductor().label(label or "L")
            elif comp == "C":
                d += elm.Capacitor().label(label or "C")
            else:
                d += elm.Line().label(label or comp)
        d += elm.Line().right()
        d += elm.Line().down()
        d += elm.Line().left()
        d += elm.Line().up()
        buf = BytesIO()
        d.save(buf, dpi=160, fmt="png")
        return buf.getvalue()


def generate_circuit(payload: Dict) -> bytes:
    ctype = (payload.get("type") or "").lower()
    if ctype == "series":
        return render_series(payload.get("components", []), payload.get("labels"))
    return _placeholder("Unsupported circuit type. Use type=series for now.")

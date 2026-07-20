"""Replace runtime MathJax equations with embedded SVGs in the HTML guide.

The notebook keeps editable TeX. The standalone HTML becomes genuinely standalone:
it does not need a CDN, JavaScript, or network access to display mathematics.
"""

from __future__ import annotations

import argparse
import base64
from html import escape, unescape
from io import BytesIO
from pathlib import Path
import re

from matplotlib.font_manager import FontProperties
from matplotlib.mathtext import math_to_image


MATHJAX_BLOCK = re.compile(
    r"<!-- Load mathjax -->.*?<!-- End of mathjax configuration -->",
    flags=re.DOTALL | re.IGNORECASE,
)
DISPLAY_MATH = re.compile(
    r"<p>\s*\$\$\s*(.*?)\s*\$\$\s*</p>", flags=re.DOTALL
)
TABLE_CELL = re.compile(r"(<td\b[^>]*>)(.*?)(</td>)", flags=re.DOTALL)
INLINE_MATH = re.compile(r"\$(.+?)\$", flags=re.DOTALL)


def normalized_tex(source: str) -> str:
    return " ".join(unescape(source).strip().split())


def equation_image(source: str, *, display: bool) -> str:
    tex = normalized_tex(source)
    buffer = BytesIO()
    math_to_image(
        tex,
        buffer,
        format="svg",
        dpi=180,
        prop=FontProperties(size=17 if display else 13),
    )
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    css_class = "static-math static-math-display" if display else "static-math static-math-inline"
    return (
        f'<img class="{css_class}" '
        f'src="data:image/svg+xml;base64,{encoded}" '
        f'alt="{escape(tex, quote=True)}" loading="eager"/>'
    )


def render(path: Path) -> tuple[int, int]:
    html = path.read_text(encoding="utf-8")
    if 'name="static-math-renderer"' in html:
        return 0, 0

    display_count = 0
    inline_count = 0

    def replace_display(match: re.Match[str]) -> str:
        nonlocal display_count
        display_count += 1
        return f'<div class="static-math-container">{equation_image(match.group(1), display=True)}</div>'

    html, mathjax_count = MATHJAX_BLOCK.subn("", html, count=1)
    if mathjax_count != 1:
        raise RuntimeError("expected exactly one MathJax runtime block")
    html = DISPLAY_MATH.sub(replace_display, html)

    def replace_cell(match: re.Match[str]) -> str:
        nonlocal inline_count

        def replace_inline(inline_match: re.Match[str]) -> str:
            nonlocal inline_count
            inline_count += 1
            return equation_image(inline_match.group(1), display=False)

        inner = INLINE_MATH.sub(replace_inline, match.group(2))
        return match.group(1) + inner + match.group(3)

    html = TABLE_CELL.sub(replace_cell, html)
    style = """
<meta name="static-math-renderer" content="matplotlib-mathtext-svg"/>
<style id="static-math-style">
.static-math-container { text-align: center; margin: 1.1rem auto; overflow-x: auto; }
.static-math-display { display: inline-block; max-width: 100%; height: auto; }
.static-math-inline { display: inline-block; max-width: 100%; height: 1.55em; width: auto; vertical-align: -0.38em; }
</style>
"""
    html = html.replace("</head>", style + "</head>", 1)
    if display_count < 20 or inline_count < 10:
        raise RuntimeError(
            f"unexpected equation count: display={display_count}, inline={inline_count}"
        )
    path.write_text(html, encoding="utf-8", newline="\n")
    return display_count, inline_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "html",
        nargs="?",
        type=Path,
        default=Path(__file__).with_name("where_we_are.html"),
    )
    args = parser.parse_args()
    display_count, inline_count = render(args.html)
    print(
        f"Static math: {display_count} display and {inline_count} inline equations in {args.html}"
    )


if __name__ == "__main__":
    main()

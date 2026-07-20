"""Replace runtime MathJax equations with embedded SVGs in the HTML guide.

The notebook keeps editable TeX. The standalone HTML becomes genuinely standalone:
it does not need a CDN, JavaScript, or network access to display mathematics.
"""

from __future__ import annotations

import argparse
import hashlib
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
LEGACY_EQUATION_IMAGE = re.compile(
    r'<img class="static-math static-math-(display|inline)" '
    r'src="data:image/svg\+xml;base64,([^"]+)" '
    r'alt="([^"]*)" loading="eager"/>'
)
INLINE_EQUATION_SVG = re.compile(
    r'(<svg class="static-math static-math-(display|inline)" '
    r'role="img" aria-label="([^"]*)" .*?</svg>)',
    flags=re.DOTALL,
)


def normalized_tex(source: str) -> str:
    tex = " ".join(unescape(source).strip().split())
    # MathJax accepts unbraced alphabet switches such as ``\mathcal A`` while
    # Matplotlib MathText requires an explicit group.
    tex = re.sub(
        r"\\(mathcal|mathbb|mathbf)\s+([A-Za-z])",
        r"\\\1{\2}",
        tex,
    )
    tex = re.sub(r"\\(le|ge|ne)(?![A-Za-z])", r"\\\1q", tex)
    tex = re.sub(
        r"\\xrightarrow\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}",
        r"\\overset{\1}{\\longrightarrow}",
        tex,
    )
    return (
        tex.replace(r"\land", r"\wedge")
        .replace(r"\lor", r"\vee")
        .replace(r"\lVert", r"\Vert")
        .replace(r"\rVert", r"\Vert")
        # MathText has no underbrace; retaining the labelled subscript preserves
        # the explanatory decomposition without displaying raw TeX.
        .replace(r"\underbrace", "")
    )


def write_html(path: Path, html: str) -> None:
    normalized = "\n".join(line.rstrip() for line in html.splitlines()) + "\n"
    path.write_text(normalized, encoding="utf-8", newline="\n")


def namespace_ids(element: str, namespace_key: str) -> str:
    namespace = "eq" + hashlib.sha256(namespace_key.encode("utf-8")).hexdigest()[:12] + "_"
    identifiers = sorted(
        set(re.findall(r'\bid="([^"]+)"', element)), key=len, reverse=True
    )
    for identifier in identifiers:
        element = element.replace(f"#{identifier}", f"#{namespace}{identifier}")
        element = element.replace(
            f'id="{identifier}"', f'id="{namespace}{identifier}"'
        )
    return element


def inline_svg(svg: str, *, css_class: str, label: str, namespace_key: str) -> str:
    match = re.search(r"<svg\b.*?</svg>", svg, flags=re.DOTALL)
    if not match:
        raise RuntimeError("MathText did not produce an SVG element")
    element = match.group(0)
    element = namespace_ids(element, namespace_key)
    attributes = (
        f'class="{css_class}" role="img" '
        f'aria-label="{escape(label, quote=True)}" '
    )
    return element.replace("<svg ", "<svg " + attributes, 1)


def equation_image(source: str, *, display: bool, namespace_key: str) -> str:
    tex = normalized_tex(source)
    buffer = BytesIO()
    math_to_image(
        f"${tex}$",
        buffer,
        format="svg",
        dpi=180,
        prop=FontProperties(size=17 if display else 13),
    )
    css_class = "static-math static-math-display" if display else "static-math static-math-inline"
    return inline_svg(
        buffer.getvalue().decode("utf-8"),
        css_class=css_class,
        label=tex,
        namespace_key=namespace_key,
    )


def render(path: Path) -> tuple[int, int]:
    html = path.read_text(encoding="utf-8")
    if 'content="matplotlib-mathtext-inline-svg-v3"' in html:
        write_html(path, html)
        return 0, 0

    if (
        'content="matplotlib-mathtext-inline-svg-v2"' in html
        or 'content="matplotlib-mathtext-inline-svg"' in html
    ):
        display_count = 0
        inline_count = 0

        def rerender_existing(match: re.Match[str]) -> str:
            nonlocal display_count, inline_count
            _element, kind, label = match.groups()
            if kind == "display":
                display_count += 1
                ordinal = display_count
            else:
                inline_count += 1
                ordinal = inline_count
            return equation_image(
                unescape(label),
                display=kind == "display",
                namespace_key=f"{kind}-{ordinal}",
            )

        html = INLINE_EQUATION_SVG.sub(rerender_existing, html)
        html = html.replace(
            'content="matplotlib-mathtext-inline-svg-v2"',
            'content="matplotlib-mathtext-inline-svg-v3"',
            1,
        )
        html = html.replace(
            'content="matplotlib-mathtext-inline-svg"',
            'content="matplotlib-mathtext-inline-svg-v3"',
            1,
        )
        if display_count < 20 or inline_count < 10:
            raise RuntimeError(
                f"unexpected namespaced equation count: display={display_count}, inline={inline_count}"
            )
        write_html(path, html)
        return display_count, inline_count

    if 'content="matplotlib-mathtext-svg"' in html:
        import base64

        display_count = 0
        inline_count = 0

        def migrate_image(match: re.Match[str]) -> str:
            nonlocal display_count, inline_count
            kind, encoded, label = match.groups()
            if kind == "display":
                display_count += 1
            else:
                inline_count += 1
            base64.b64decode(encoded, validate=True)
            return equation_image(
                unescape(label),
                display=kind == "display",
                namespace_key=f"{kind}-{display_count if kind == 'display' else inline_count}",
            )

        html = LEGACY_EQUATION_IMAGE.sub(migrate_image, html)
        html = html.replace(
            'content="matplotlib-mathtext-svg"',
            'content="matplotlib-mathtext-inline-svg-v3"',
            1,
        )
        if display_count < 20 or inline_count < 10:
            raise RuntimeError(
                f"unexpected migrated equation count: display={display_count}, inline={inline_count}"
            )
        write_html(path, html)
        return display_count, inline_count

    display_count = 0
    inline_count = 0

    def replace_display(match: re.Match[str]) -> str:
        nonlocal display_count
        display_count += 1
        return f'<div class="static-math-container">{equation_image(match.group(1), display=True, namespace_key=f"display-{display_count}")}</div>'

    html, mathjax_count = MATHJAX_BLOCK.subn("", html, count=1)
    if mathjax_count != 1:
        raise RuntimeError("expected exactly one MathJax runtime block")
    html = DISPLAY_MATH.sub(replace_display, html)

    def replace_cell(match: re.Match[str]) -> str:
        nonlocal inline_count

        def replace_inline(inline_match: re.Match[str]) -> str:
            nonlocal inline_count
            inline_count += 1
            return equation_image(
                inline_match.group(1),
                display=False,
                namespace_key=f"inline-{inline_count}",
            )

        inner = INLINE_MATH.sub(replace_inline, match.group(2))
        return match.group(1) + inner + match.group(3)

    html = TABLE_CELL.sub(replace_cell, html)
    style = """
<meta name="static-math-renderer" content="matplotlib-mathtext-inline-svg-v3"/>
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
    write_html(path, html)
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

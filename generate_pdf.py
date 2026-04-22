from pathlib import Path
import re
import textwrap


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "ai-code-agents-poc-plan.md"
OUTPUT = ROOT / "ai-code-agents-poc-plan.pdf"


def markdown_to_lines(text: str) -> list[str]:
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)
    lines = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line:
            lines.append("")
            continue
        if line.startswith("#"):
            line = line.lstrip("#").strip().upper()
            lines.append(line)
            lines.append("")
            continue
        wrapped = textwrap.wrap(
            line,
            width=92,
            replace_whitespace=False,
            drop_whitespace=True,
            break_long_words=False,
            break_on_hyphens=False,
        )
        lines.extend(wrapped or [""])
    return lines


def split_pages(lines: list[str], lines_per_page: int = 48) -> list[list[str]]:
    pages = []
    current = []
    for line in lines:
        if len(current) >= lines_per_page:
            pages.append(current)
            current = []
        current.append(line)
    if current:
        pages.append(current)
    return pages


def pdf_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_stream(page_lines: list[str]) -> bytes:
    y_start = 760
    leading = 14
    parts = ["BT", "/F1 10 Tf", f"50 {y_start} Td", f"{leading} TL"]
    for line in page_lines:
        safe = pdf_escape(line)
        parts.append(f"({safe}) Tj")
        parts.append("T*")
    parts.append("ET")
    return "\n".join(parts).encode("latin-1", errors="replace")


def make_pdf(pages: list[list[str]]) -> bytes:
    objects: list[bytes] = []

    def add_object(data: bytes) -> int:
        objects.append(data)
        return len(objects)

    font_id = add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    page_ids = []
    content_ids = []
    placeholder_page = b""
    for page in pages:
        stream = build_stream(page)
        content_obj = (
            f"<< /Length {len(stream)} >>\nstream\n".encode("latin-1")
            + stream
            + b"\nendstream"
        )
        content_ids.append(add_object(content_obj))
        page_ids.append(add_object(placeholder_page))

    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    pages_id = add_object(
        f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode("latin-1")
    )
    catalog_id = add_object(f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode("latin-1"))

    for idx, page_id in enumerate(page_ids):
        content_id = content_ids[idx]
        page_obj = (
            f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> "
            f"/Contents {content_id} 0 R >>"
        ).encode("latin-1")
        objects[page_id - 1] = page_obj

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{i} 0 obj\n".encode("latin-1"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")

    xref_start = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("latin-1"))

    trailer = (
        f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\n"
        f"startxref\n{xref_start}\n%%EOF\n"
    )
    pdf.extend(trailer.encode("latin-1"))
    return bytes(pdf)


def main() -> None:
    lines = markdown_to_lines(SOURCE.read_text(encoding="utf-8"))
    pages = split_pages(lines)
    OUTPUT.write_bytes(make_pdf(pages))
    print(OUTPUT)


if __name__ == "__main__":
    main()

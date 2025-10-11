import pathlib
from typing import List, Tuple
import bs4
import html
import typer


app = typer.Typer(name="Tydier")


def extract_bookmarks(soup: bs4.BeautifulSoup) -> List[Tuple[str, str]]:
    """Return a flat list of (title, href) tuples for each bookmark found.

    This finds all <a href="...">text</a> entries in the HTML and
    returns their text and href. It intentionally flattens any nested
    structures and ignores anchors without hrefs.
    """
    bookmarks: List[Tuple[str, str]] = []
    for a in soup.find_all("a", href=True):
        title = a.get_text(strip=True)
        href = a["href"]
        bookmarks.append((title, href))
    return bookmarks


def write_bookmarks_netscape(output_path: pathlib.Path, bookmarks: List[Tuple[str, str]]) -> None:
    """Write bookmarks in the common Netscape bookmarks HTML format.

    bookmarks: list of (title, href) tuples.
    """
    header = """<!DOCTYPE NETSCAPE-Bookmark-file-1>
        <META HTTP-EQUIV=Content-Type CONTENT=\"text/html; charset=UTF-8\">
        <TITLE>Bookmarks</TITLE>
        <H1>Bookmarks</H1>
        <DL><p>
        """
    footer = "</DL><p>\n"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(header)
        for title, href in bookmarks:
            safe_title = (title or href).replace('\n', ' ').strip()
            # Escape title and href for HTML safety
            esc_title = html.escape(safe_title)
            esc_href = html.escape(href, quote=True)
            # Write a single DT entry per bookmark. No add_date or other attrs.
            f.write(f"    <DT><A HREF=\"{esc_href}\">{esc_title}</A>\n")
        f.write(footer)


@app.command()
def process(
    input_file: str = typer.Argument(help="Path to input HTML file"),
    output: str = typer.Option("output.html", "-o", "--output", help="Output bookmarks.html path")
) -> None:
    """Process INPUT_FILE, extract bookmarks, print count, and write placeholder output.

    Example: python main.py input.html --output output.html
    """
    in_path = pathlib.Path(input_file)
    out_path = pathlib.Path(output)

    if not in_path.exists():
        typer.secho(f"Input file does not exist: {in_path}", fg=typer.colors.RED)
        raise typer.Exit(code=2)

    try:
        text = in_path.read_text(encoding="utf-8")
    except Exception as exc:
        typer.secho(f"Failed to read input file: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=3)

    soup = bs4.BeautifulSoup(text, "html.parser")
    bookmarks = extract_bookmarks(soup)

    typer.secho(f"Found {len(bookmarks)} bookmarks.", fg=typer.colors.GREEN)

    # Write the extracted bookmarks in Netscape bookmark HTML format.
    try:
        write_bookmarks_netscape(out_path, bookmarks)
        typer.secho(f"Bookmarks written to {out_path}", fg=typer.colors.BLUE)
    except Exception as exc:
        typer.secho(f"Failed to write output file: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=4)


if __name__ == "__main__":
    app()
import pathlib
import sys
import tempfile

import bs4

# Ensure the project root is on sys.path so `main` can be imported when tests run
proj_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(proj_root))

from main import extract_bookmarks, write_bookmarks_netscape


def test_extract_and_write(tmp_path: pathlib.Path):
    html = """
    <html>
      <body>
        <a href="https://example.com">Example</a>
        <div>
          <a href="/local">Local</a>
        </div>
      </body>
    </html>
    """
    soup = bs4.BeautifulSoup(html, "html.parser")
    bookmarks = extract_bookmarks(soup)
    assert ("Example", "https://example.com") in bookmarks
    assert ("Local", "/local") in bookmarks

    out = tmp_path / "out.html"
    write_bookmarks_netscape(out, bookmarks)
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "https://example.com" in text
    assert "Local" in text

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
    <!DOCTYPE NETSCAPE-Bookmark-file-1>
    <!-- This is an automatically generated file.
        It will be read and overwritten.
        DO NOT EDIT! -->
    <META HTTP-EQUIV="Content-Type" CONTENT="text/html; charset=UTF-8">
    <TITLE>Bookmarks</TITLE>
    <H1>Bookmarks</H1>
    <DL><p>
        <DT><H3 ADD_DATE="1757525265" LAST_MODIFIED="0" PERSONAL_TOOLBAR_FOLDER="true">Bookmarks</H3>
        <DL><p>
          <DT><A HREF="https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf?mcp_token=eyJwaWQiOjE3MjIwODgsInNpZCI6MTA0ODY0MzA5MCwiYXgiOiI5ZTQyYTg3NmMwMjg0MzBmMWM3ZmY3NjgxNmY4ZDE3ZiIsInRzIjoxNzUyODQ0NjYzLCJleHAiOjE3NTUyNjM4NjN9.bMLlLtW44gdsC59Ue0RRib1FUxB31nwceQ-PXFNx0mM" ADD_DATE="1752844670">A Practical Guide to Building Agents</A>
          <DT><A HREF="https://foundersignal.app/" ADD_DATE="1754059977" ICON="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAADGElEQVR4nFxTS2hTQRS9M/P+iUk1Fmu1FRoTUMEPUhAxVVE3iuJKpF2pC/+6UJGuBMEPgh/cCCKuRBARuimKLoTSGrpyYVSQ4oemqbZqmsYk7+W9N9c7sVXpwDAw995z7j1nhsOfxeDf2mxbxsN5jvM5Fo0GtL2obX0whbir6/p6mLP+L4xFHetaNBo9YDsRQ4YhBEEAjPFGkkQJpdLUdL0e3K4HwUW68mcB1I5FbOuJ7/vbDMNAOmXdD7gK2LYFnHN0HAellIIxBsXi1AM/DA9SOBAKxbGsW3S/r729vd7a2spbWhaLjmQHNDc3w+TEhALgrutypEYs0wyIZF3ge9VQ4iC3NK2LC3aoVnPDVCqtXbp8hScSC5CKWTqdZojIypUquq6Hdc9j0+WyiMViMhKJniXu5ZrQtaMMmU7oQX4szwuFAj57/oImp9lohoXUxYYVK5hEhLe5HExPl0mLEgFEEpVKtVvlbUSgKAA3dB0U42wxMUFPTw9wocHKlasglUphGAZA40C97qNu6Ls0ym0lYZhSjFRniYUJyHRlgNBh65bNsGbtOrhx8xZkXw0BCclM02yQuG4NaL75ikyxo+pBKewReqEwBsWfP2BwcAiSySTs3bMbqq4HtVpNCdpIDslmdSriL41qBcQY0ow4MvIR8vk8ZIeH4cjhw3Ds+Ak4feokVGuuykQiUlsxjnMS56UaQQXIHiAHmGUaoGkaLFncArncG+ju7oZMpgt7e3tZpdrogikcKbGP+6G8QyAVXdPY/KYmHB3NYyKRQNu2kd6FmhtKU0U8dPAA25TZBGvXrJbUidI8X/O8R5xe3WuUeN00dD4wMBCeO3tGTk5OKvsYdYHqDRAYK5V/4cS3iXAZgYZSck2IC4RSaDhWdd1L1M4D6ks3DUNQi/Lr+Lhc2taOqWQH4aPcsX2bpHuRzWaFpvGrU+Xy/Rm3/y7hWMZ5xsUpztgiz/NgaVsbdHZ2Qjweh3hTEzzt7/+Ue/f+MuXem/lDyOZ+T/I5LRjuF1zbSV4vUD7F58XHvxeLfRR+TF6NzfzhhnW/AQAA//81xbeLAAAABklEQVQDAFnsbSkyfo8EAAAAAElFTkSuQmCC">FounderSignal - Validate Your Startup Idea in 72 Hours | FounderSignal</A>
          <DT><A HREF="https://learn.founderz.com/" ADD_DATE="1752845380" ICON="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAABnklEQVR4nHSSv0vDQBTH37tLq0IrVlrt4OIizoKLQweLKIgoLi7OTv4vLiJYUQr9GwQFHXSp4CjqpLapWkprK1XsjyR3vsvVmAZ9hEsued/v+7yXM9YWroBCAiAgghCSMaQVafNXMH2TCFLKTltQnl4dR+pLOGRGH6XONPSN3DpdkV5MLK8mT48r97cfW9uTkqpxNAutbMZ0GXwCVYvhynpyajo6OMTKr+2Z2RH9fmx8IJspUgUN6QmUQXbfTKXj+ct6pdzZ23kkHm5gvWYRjteSJ6C9rFW71/kG2ZfM9tPDi7KREA6zSPQXpPfEObw37PRSYmNz4uK8mjssRYcN4QByaLxZjXqXc3RNfwQuoZomxVxqNDUfpwchyAip49yRGYuFaRvsQQeh25aQ0BPQFgGDSLonxYB4dlI72C1Eolxb2pZUeAKCY3XBVB2rKz6btvvXtZeauDso8PfQQ3IPBdA0CcY7HN5P6EOit6EQIzNaPb0fuE9AxkaIPZdadzfNYuGLG6xf4hPr06qLah5K9UMHgvmQVB6Azpb/nG74BgAA//+Ntu+SAAAABklEQVQDAOEw3KeZ9RT7AAAAAElFTkSuQmCC">Founderz | Online Business School</A>
        </DL><p>
    </DL><p>
    """
    soup = bs4.BeautifulSoup(html, "html.parser")
    bookmarks = extract_bookmarks(soup)
    assert ("A Practical Guide to Building Agents", "https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf?mcp_token=eyJwaWQiOjE3MjIwODgsInNpZCI6MTA0ODY0MzA5MCwiYXgiOiI5ZTQyYTg3NmMwMjg0MzBmMWM3ZmY3NjgxNmY4ZDE3ZiIsInRzIjoxNzUyODQ0NjYzLCJleHAiOjE3NTUyNjM4NjN9.bMLlLtW44gdsC59Ue0RRib1FUxB31nwceQ-PXFNx0mM") in bookmarks
    assert ("FounderSignal - Validate Your Startup Idea in 72 Hours | FounderSignal", "https://foundersignal.app/") in bookmarks

    out = tmp_path / "out.html"
    write_bookmarks_netscape(out, bookmarks)
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf" in text
    assert "FounderSignal" in text

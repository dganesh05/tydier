"""
Future Ideas:
- Add a way for Groq to compare all folder headers to make them distinct
"""

import pathlib
from typing import List, Tuple
import bs4
import html
import typer
import requests
import pandas as pd
import io
import pdfplumber
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import os
import dotenv
from groq import Groq

dotenv.load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

model = SentenceTransformer("all-MiniLM-L6-v2")
app = typer.Typer(name="Tydier")
client = Groq(api_key=GROQ_API_KEY)
groq_model = "llama-3.1-8b-instant"


N_CLUSTERS = 10

#================================================================
# Data Collection and Processing
#================================================================

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

def _fetch_html(url: str, timeout_seconds: int = 8) -> str:
    """Fetch raw HTML from a URL with basic headers and timeouts.

    Returns an empty string on any failure to be robust in batch processing.
    """
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=headers, timeout=timeout_seconds, allow_redirects=True)
        if resp.status_code >= 200 and resp.status_code < 300:
            # Best-effort encoding handling
            resp.encoding = resp.apparent_encoding or resp.encoding
            return resp.text or ""
    except Exception:
        # Intentionally swallow errors to avoid breaking the whole run
        return ""
    return ""

def _fetch_response(url: str, timeout_seconds: int = 8):
    """Fetch a Response object for detailed inspection. Returns None on failure."""
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=headers, timeout=timeout_seconds, allow_redirects=True)
        if resp.status_code >= 200 and resp.status_code < 300:
            return resp
    except Exception:
        return None
    return None

def _extract_pdf_text_from_bytes(blob: bytes, max_pages: int = 2, max_chars: int = 1000) -> str:
    """Extract up to max_chars of text from first max_pages of a PDF byte blob."""
    if not blob:
        return ""
    try:
        with pdfplumber.open(io.BytesIO(blob)) as pdf:
            result = []
            for page in pdf.pages[:max_pages]:
                page_text = page.extract_text() or ""
                if page_text:
                    result.append(page_text)
                if sum(len(s) for s in result) >= max_chars:
                    break
            text = "\n".join(result)
            return text[:max_chars].strip()
    except Exception:
        return ""

def _extract_description_from_html(html_text: str) -> str:
    """Extract description from HTML.

    Priority order:
    1) <meta name="description" content="...">
    2) <meta property="og:description" content="...">
    3) <meta name="twitter:description" content="...">
    4) Fallback to first lines of the body text
    """
    if not html_text:
        return ""

    soup = bs4.BeautifulSoup(html_text, "html.parser")

    # Try common meta descriptions
    meta = (
        soup.find("meta", attrs={"name": "description"})
        or soup.find("meta", attrs={"property": "og:description"})
        or soup.find("meta", attrs={"name": "twitter:description"})
    )
    if meta and meta.get("content"):
        return meta.get("content", "").strip()

    # Fallback: extract first lines of visible body text
    body = soup.body or soup
    text = body.get_text(" ", strip=True)
    # Normalize whitespace and take an informative snippet
    snippet = (text or "").strip()
    if not snippet:
        return ""

    # Prefer a couple of sentences if possible; otherwise first ~300 chars
    period_idx = snippet.find(".")
    if period_idx != -1 and period_idx < 200:
        # Take up to two sentences
        second_period_idx = snippet.find(".", period_idx + 1)
        cutoff = second_period_idx if second_period_idx != -1 else period_idx
        return snippet[: cutoff + 1].strip()[:400]

    return snippet[:400]

def _mine_url_description(url: str) -> str:
    """Best-effort mining of a page description for a URL.

    Returns an empty string on failure.
    """
    # Use detailed path so PDFs and non-HTML are handled consistently
    details = _mine_url_details(url)
    return details.get("description", "")

def _mine_url_details(url: str) -> dict:
    """Return detailed mining info for inspection and debugging.

    Keys:
      - html_length
      - meta_description
      - og_description
      - twitter_description
      - body_snippet
      - description (chosen)
      - description_source (meta|og|twitter|body|unknown)
    """
    info = {
        "html_length": 0,
        "meta_description": "",
        "og_description": "",
        "twitter_description": "",
        "body_snippet": "",
        "description": "",
        "description_source": "unknown",
    }
    # Local file:// PDF support (e.g., file:///C:/path/file.pdf)
    if url.lower().startswith("file://"):
        local_path = url[7:] if url.lower().startswith("file:///") else url[7:]
        try:
            # Try PDF extraction directly
            text = ""
            try:
                with pdfplumber.open(local_path) as pdf:
                    result = []
                    for page in pdf.pages[:2]:
                        page_text = page.extract_text() or ""
                        if page_text:
                            result.append(page_text)
                    text = "\n".join(result)[:1000].strip()
            except Exception:
                text = ""
            info["content_type"] = "application/pdf" if local_path.lower().endswith(".pdf") else "file"
            info["html_length"] = 0
            info["body_snippet"] = text
            if text:
                info["description"] = text
                info["description_source"] = "pdf"
            else:
                info["description"] = ""
                info["description_source"] = "non_html"
            return info
        except Exception:
            return info

    resp = _fetch_response(url)
    if resp is None:
        return info
    content_type = (resp.headers.get("Content-Type") or "").lower()
    info["content_type"] = content_type

    # Only attempt HTML/text parsing for text-like responses
    if ("text/html" not in content_type) and ("text/" not in content_type):
        # PDF handling
        if "application/pdf" in content_type or url.lower().endswith(".pdf"):
            snippet = _extract_pdf_text_from_bytes(resp.content)
            info["html_length"] = len(resp.content or b"")
            info["body_snippet"] = snippet
            if snippet:
                info["description"] = snippet
                info["description_source"] = "pdf"
            else:
                info["description"] = ""
                info["description_source"] = "non_html"
            return info

        # Other non-HTML types: return with non_html
        info["html_length"] = len(resp.content or b"")
        info["description"] = ""
        info["description_source"] = "non_html"
        info["body_snippet"] = ""
        return info

    # Treat as HTML
    resp.encoding = resp.apparent_encoding or resp.encoding
    html_text = resp.text or ""
    info["html_length"] = len(html_text)

    soup = bs4.BeautifulSoup(html_text, "html.parser")
    m_meta = soup.find("meta", attrs={"name": "description"})
    m_og = soup.find("meta", attrs={"property": "og:description"})
    m_tw = soup.find("meta", attrs={"name": "twitter:description"})
    if m_meta and m_meta.get("content"):
        info["meta_description"] = (m_meta.get("content") or "").strip()
    if m_og and m_og.get("content"):
        info["og_description"] = (m_og.get("content") or "").strip()
    if m_tw and m_tw.get("content"):
        info["twitter_description"] = (m_tw.get("content") or "").strip()

    # Body snippet
    body = soup.body or soup
    body_text = body.get_text(" ", strip=True)
    info["body_snippet"] = (body_text or "").strip()[:400]

    # Choose description consistent with _extract_description_from_html
    if info["meta_description"]:
        info["description"] = info["meta_description"]
        info["description_source"] = "meta"
    elif info["og_description"]:
        info["description"] = info["og_description"]
        info["description_source"] = "og"
    elif info["twitter_description"]:
        info["description"] = info["twitter_description"]
        info["description_source"] = "twitter"
    else:
        info["description"] = info["body_snippet"]
        info["description_source"] = "body" if info["body_snippet"] else "unknown"

    return info

# Can remove this function and just use the write_clustered_bookmarks_netscape function in produciton
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

def write_clustered_bookmarks_netscape(output_path: pathlib.Path, clusters: List[List[Tuple[str, str]]], folder_names: List[str] = None) -> None:
    """Write clustered bookmarks in the common Netscape bookmarks HTML format.

    clusters: list of clusters, where each cluster is a list of (title, href) tuples.
    folder_names: optional list of folder names for each cluster.
    """
    header = """<!DOCTYPE NETSCAPE-Bookmark-file-1>
        <META HTTP-EQUIV=Content-Type CONTENT=\"text/html; charset=UTF-8\">
        <TITLE>Clustered Bookmarks</TITLE>
        <H1>Clustered Bookmarks</H1>
        <DL><p>
        """
    footer = "</DL><p>\n"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(header)
        folder_index = 0
        for i, cluster in enumerate(clusters):
            if cluster:  # Only write non-empty clusters
                # Write cluster header with folder name if available
                if folder_names and folder_index < len(folder_names):
                    cluster_title = html.escape(folder_names[folder_index])
                    f.write(f"    <DT><H3>{cluster_title} ({len(cluster)} bookmarks)</H3>\n")
                    folder_index += 1
                else:
                    f.write(f"    <DT><H3>Cluster {i+1} ({len(cluster)} bookmarks)</H3>\n")
                f.write("    <DL><p>\n")
                
                # Write bookmarks in this cluster
                for title, href in cluster:
                    safe_title = (title or href).replace('\n', ' ').strip()
                    # Escape title and href for HTML safety
                    esc_title = html.escape(safe_title)
                    esc_href = html.escape(href, quote=True)
                    f.write(f"        <DT><A HREF=\"{esc_href}\">{esc_title}</A>\n")
                
                f.write("    </DL><p>\n")
        f.write(footer)

#================================================================
# Embedding Model
#================================================================

def embed_bookmarks(bookmarks: List[Tuple[str, str]], enable_web_mining: bool = False) -> List[Tuple[str, List[float], str]]:
    """Embed bookmark titles using the sentence transformer model.

    When enable_web_mining=True, fetch each URL and append a description
    (meta description or first lines of body) to enrich embeddings.
    """
    embeddings: List[Tuple[str, List[float], str]] = []
    for title, href in bookmarks:
        enriched_text = title
        if enable_web_mining:
            # Mine description to enrich the text we embed; remain robust if mining fails
            description = _mine_url_description(href)
            if description:
                enriched_text = f"{title}. {description}"
        vector = model.encode(enriched_text).tolist()
        embeddings.append((title, vector, href))
    return embeddings

#================================================================
# Clustering
#================================================================

def cluster_bookmarks(embedded_bookmarks: List[Tuple[str, List[float], str]], n_clusters: int = 10) -> List[List[Tuple[str, str]]]:
    """Cluster bookmarks using the sentence transformer model."""
    if len(embedded_bookmarks) < n_clusters:
        # If we have fewer bookmarks than clusters, return each bookmark as its own cluster
        return [[(title, href)] for title, _, href in embedded_bookmarks]
    
    embeddings = [embedding for title, embedding, href in embedded_bookmarks]
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    cluster_labels = clustering.fit_predict(embeddings)
    
    # Group bookmarks by cluster
    clusters = [[] for _ in range(n_clusters)]
    for i, (title, _, href) in enumerate(embedded_bookmarks):
        cluster_id = cluster_labels[i]
        clusters[cluster_id].append((title, href))
    
    # Filter out empty clusters
    return [cluster for cluster in clusters if cluster]

#================================================================
# Folder Naming
#================================================================

def get_folder_name(titles: List[str]) -> str:
    """Get a folder name for a bookmark title."""
    if not client:
        typer.secho("Warning: GROQ_API_KEY not set, using default folder names", fg=typer.colors.YELLOW)
        return f"Cluster {len(titles)} Bookmarks"
    
    try:
        # Limit titles to avoid token limits
        limited_titles = titles[:10]  # Only use first 10 titles
        prompt = f"Generate a concise folder name (2-4 words) for bookmarks with these titles: {limited_titles}. Return only the folder name, no explanation."
        
        response = client.chat.completions.create(
            model=groq_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0.3
        )
        folder_name = response.choices[0].message.content.strip()
        # Clean up the response (remove quotes, extra text)
        folder_name = folder_name.strip('"').strip("'").strip()
        return folder_name
    except Exception as e:
        typer.secho(f"Warning: Failed to get folder name from Groq: {e}", fg=typer.colors.YELLOW)
        return f"Cluster {len(titles)} Bookmarks"
    
#================================================================
# CLI Command
#================================================================

@app.command()
def process(
    input_file: str = typer.Argument(help="Path to input HTML file"),
    output: str = typer.Option("output.html", "-o", "--output", help="Output bookmarks.html path"),
    web_mine: bool = typer.Option(True, "--web-mine/--no-web-mine", help="Enable web mining for descriptions")
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

    # Embed the bookmarks (optionally enriched by web mining)
    embedded_bookmarks = embed_bookmarks(bookmarks, enable_web_mining=web_mine)
    typer.secho(f"Embedded {len(embedded_bookmarks)} bookmarks.", fg=typer.colors.GREEN)
    
    # Cluster the bookmarks
    clusters = cluster_bookmarks(embedded_bookmarks)
    typer.secho(f"Created {len(clusters)} clusters.", fg=typer.colors.GREEN)
    
    # Display cluster information
    for i, cluster in enumerate(clusters):
        if cluster:
            typer.secho(f"  Cluster {i+1}: {len(cluster)} bookmarks", fg=typer.colors.CYAN)
            # Show first few bookmarks in each cluster as examples
            for j, (title, href) in enumerate(cluster[:3]):
                typer.secho(f"    - {title}", fg=typer.colors.WHITE)
            if len(cluster) > 3:
                typer.secho(f"    ... and {len(cluster) - 3} more", fg=typer.colors.WHITE)

    # Get a folder name for each cluster
    folder_names = []
    for i, cluster in enumerate(clusters):
        if cluster:
            folder_name = get_folder_name([title for title, href in cluster])
            typer.secho(f"  Cluster {i+1} folder name: {folder_name}", fg=typer.colors.CYAN)
            folder_names.append(folder_name)

    # Write the clustered bookmarks in Netscape bookmark HTML format.
    try:
        write_clustered_bookmarks_netscape(out_path, clusters, folder_names)
        typer.secho(f"Clustered bookmarks written to {out_path}", fg=typer.colors.BLUE)
    except Exception as exc:
        typer.secho(f"Failed to write output file: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=4)


@app.command("preview-mining")
def preview_mining(
    input_file: str = typer.Argument(help="Path to input HTML file"),
    limit: int = typer.Option(0, "--limit", min=0, help="Limit number of bookmarks to preview; 0 = all"),
    output_csv: str = typer.Option(None, "--output-csv", help="Optional path to write CSV preview")
) -> None:
    """Preview mined metadata/content as a pandas DataFrame for quick inspection."""
    in_path = pathlib.Path(input_file)
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
    if not bookmarks:
        typer.secho("No bookmarks found.", fg=typer.colors.YELLOW)
        raise typer.Exit(code=0)

    rows = []
    sample = bookmarks if limit == 0 else bookmarks[:limit]
    for title, href in sample:
        info = _mine_url_details(href)
        rows.append({
            "title": title,
            "url": href,
            "description_source": info.get("description_source", ""),
            "description": info.get("description", ""),
            "meta_description": info.get("meta_description", ""),
            "og_description": info.get("og_description", ""),
            "twitter_description": info.get("twitter_description", ""),
            "body_snippet": info.get("body_snippet", ""),
            "html_length": info.get("html_length", 0),
            "content_type": info.get("content_type", ""),
        })

    df = pd.DataFrame(rows)
    # Print concise view
    with pd.option_context("display.max_colwidth", 120, "display.width", 200):
        print(df)

    if output_csv:
        try:
            df.to_csv(output_csv, index=False)
            typer.secho(f"Wrote CSV preview to {output_csv}", fg=typer.colors.BLUE)
        except Exception as exc:
            typer.secho(f"Failed to write CSV: {exc}", fg=typer.colors.RED)

if __name__ == "__main__":
    app()
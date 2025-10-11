import pathlib
from typing import List, Tuple
import bs4
import html
import typer
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

model = SentenceTransformer("all-MiniLM-L6-v2")
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


def write_clustered_bookmarks_netscape(output_path: pathlib.Path, clusters: List[List[Tuple[str, str]]]) -> None:
    """Write clustered bookmarks in the common Netscape bookmarks HTML format.

    clusters: list of clusters, where each cluster is a list of (title, href) tuples.
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
        for i, cluster in enumerate(clusters):
            if cluster:  # Only write non-empty clusters
                # Write cluster header
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

def embed_bookmarks(bookmarks: List[Tuple[str, str]]) -> List[Tuple[str, List[float], str]]:
    """Embed bookmark titles using the sentence transformer model."""
    embeddings = [(title, model.encode(title).tolist(), href) for title, href in bookmarks]
    return embeddings

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

    # Embed the bookmarks
    embedded_bookmarks = embed_bookmarks(bookmarks)
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

    # Write the clustered bookmarks in Netscape bookmark HTML format.
    try:
        write_clustered_bookmarks_netscape(out_path, clusters)
        typer.secho(f"Clustered bookmarks written to {out_path}", fg=typer.colors.BLUE)
    except Exception as exc:
        typer.secho(f"Failed to write output file: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=4)


if __name__ == "__main__":
    app()
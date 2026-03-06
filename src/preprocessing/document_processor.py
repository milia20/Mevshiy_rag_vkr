"""
Document preprocessing module for MkDocs-based RAG pipelines.

This module:
1. Recursively scans a documentation directory for Markdown files.
2. Cleans Markdown content (removes images, HTML, buttons, annotations).
3. Preserves headers and code blocks (important for technical docs).
4. Performs semantic chunking based on Markdown headers.
5. Applies RecursiveCharacterTextSplitter for chunk size control.
6. Extracts metadata required for vector indexing.

Output format:
{
    "text": "...",
    "metadata": {
        "source": "docs/setup.md",
        "url": "/setup/",
        "headers": ["Setup", "Installation"],
        "chunk_id": "uuid",
        "doc_title": "Setup"
    }

Chunks are written to:
data/processed/chunks.jsonl
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter


def clean_markdown(content: str) -> str:
    """
    Clean Markdown while preserving useful technical information.

    Removes:
        - Image markup ![alt](url)
        - HTML tags
        - Button formatting ~~**button**~~
        - Footnotes and annotations

    Keeps:
        - Headers (#, ##, ###)
        - Code blocks
        - Regular text

    Parameters
    ----------
    content : str
        Raw markdown content.

    Returns
    -------
    str
        Cleaned markdown text.
    """

    # Remove image markdown
    content = re.sub(r"!\[.*?\]\(.*?\)", "", content)

    # Remove HTML tags
    content = re.sub(r"<[^>]+>", "", content)

    # Remove button formatting (~~**text**~~)
    content = re.sub(r"~~\*\*(.*?)\*\*~~", r"\1", content)

    # Remove footnotes [^1]
    content = re.sub(r"\[\^.*?\]", "", content)

    # Remove footnote definitions
    content = re.sub(r"\[\^.*?\]: .*", "", content)

    # Remove excessive whitespace
    content = re.sub(r"\n{3,}", "\n\n", content)

    return content.strip()


# Header-based semantic splitting
HEADER_PATTERN = re.compile(r"^(#{1,3})\s+(.*)")


def split_by_headers(text: str) -> List[Tuple[List[str], str]]:
    """
    Split markdown text by headers while preserving hierarchy.

    Example output:
        [
            (["Setup"], "text under H1"),
            (["Setup", "Installation"], "text under H2")
        ]

    Parameters
    ----------
    text : str

    Returns
    -------
    list[tuple[list[str], str]]
        List of sections with header hierarchy.
    """

    sections: List[Tuple[List[str], str]] = []
    headers: List[str] = []
    buffer: List[str] = []

    for line in text.splitlines():

        match = HEADER_PATTERN.match(line)

        if match:
            if buffer:
                sections.append((headers.copy(), "\n".join(buffer).strip()))
                buffer = []

            level = len(match.group(1))
            title = match.group(2).strip()

            headers = headers[: level - 1]
            headers.append(title)

        buffer.append(line)

    if buffer:
        sections.append((headers.copy(), "\n".join(buffer).strip()))

    return sections


def scan_markdown_files(directory: Path) -> Iterable[Path]:
    """
    Recursively find all markdown files.

    Parameters
    ----------
    directory : Path

    Yields
    ------
    Path
    """

    yield from directory.rglob("*.md")


def convert_path_to_url(file_path: Path, docs_root: Path) -> str:
    """
    Convert file path to documentation URL.

    Example:
        docs/setup.md -> /setup/

    Parameters
    ----------
    file_path : Path
    docs_root : Path

    Returns
    -------
    str
    """

    relative = file_path.relative_to(docs_root)
    url = "/" + str(relative).replace(".md", "") + "/"
    url = url.replace("index/", "")

    return url


def create_text_splitter(
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> RecursiveCharacterTextSplitter:
    """
    Create RecursiveCharacterTextSplitter.

    Parameters
    ----------
    chunk_size : int
    chunk_overlap : int

    Returns
    -------
    RecursiveCharacterTextSplitter
    """

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )


def process_markdown_file(
    path: Path,
    docs_root: Path,
    splitter: RecursiveCharacterTextSplitter,
) -> List[Dict]:
    """
    Process a single markdown document.

    Parameters
    ----------
    path : Path
    docs_root : Path
    splitter : RecursiveCharacterTextSplitter

    Returns
    -------
    list[dict]
    """

    text = path.read_text(encoding="utf-8")

    cleaned = clean_markdown(text)

    sections = split_by_headers(cleaned)

    chunks: List[Dict] = []

    for headers, section_text in sections:

        split_chunks = splitter.split_text(section_text)

        for chunk in split_chunks:

            chunk_id = str(uuid.uuid4())

            chunks.append(
                {
                    "text": chunk,
                    "metadata": {
                        "source": str(path.relative_to(docs_root)),
                        "url": convert_path_to_url(path, docs_root),
                        "headers": headers,
                        "chunk_id": chunk_id,
                        "doc_title": headers[0] if headers else path.stem,
                    },
                }
            )

    return chunks


def process_docs(
    docs_dir: str,
    output_path: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> None:
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    docs_dir : str
        Directory containing markdown docs
    output_path : str
        JSONL output file
    chunk_size : int
    chunk_overlap : int
    """

    docs_root = Path(docs_dir)
    output_file = Path(output_path)

    splitter = create_text_splitter(chunk_size, chunk_overlap)

    all_chunks: List[Dict] = []

    for md_file in scan_markdown_files(docs_root):

        chunks = process_markdown_file(
            md_file,
            docs_root,
            splitter,
        )

        all_chunks.extend(chunks)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"Saved {len(all_chunks)} chunks -> {output_file}")


if __name__ == "__main__":

    DOCS_DIR = r"D:\P_work\Rag-VKR\fastapi_doc\en\docs" #"docs"
    OUTPUT_FILE = "data/processed/chunks_en.jsonl"

    process_docs(
        docs_dir=DOCS_DIR,
        output_path=OUTPUT_FILE,
        chunk_size=512,
        chunk_overlap=50,
    )
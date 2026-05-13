"""
Document preprocessing module for MkDocs-based RAG pipelines - IMPROVED VERSION

This module:
1. Recursively scans a documentation directory for Markdown files.
2. Cleans Markdown content (removes images, HTML, buttons, annotations, links, tables).
3. Preserves headers and code blocks (important for technical docs).
4. Performs semantic chunking based on Markdown headers.
5. Applies RecursiveCharacterTextSplitter for chunk size control.
6. Extracts metadata required for vector indexing.
7. Includes error handling, logging, and input validation.

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
import logging
import re
import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def clean_markdown(content: str) -> str:
    """
    Clean Markdown while preserving useful technical information.

    Removes:
        - Image markup ![alt](url)
        - HTML tags
        - Button formatting ~~**button**~~
        - Footnotes and annotations
        - Inline links [text](url)
        - Tables
        - Inline code `code`

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

    # Remove inline links [text](url) - keep text only
    content = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", content)

    # Remove tables (simplified approach)
    content = re.sub(r"\|.*?\|", "", content, flags=re.MULTILINE)
    content = re.sub(r"\|[-\s|]+\|", "", content)

    # Remove inline code `code` - keep code content
    content = re.sub(r"`([^`]+)`", r"\1", content)

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
    Recursively find all markdown files with error handling.

    Parameters
    ----------
    directory : Path

    Yields
    ------
    Path
    """
    if not directory.exists():
        logger.error(f"Directory does not exist: {directory}")
        return

    if not directory.is_dir():
        logger.error(f"Path is not a directory: {directory}")
        return

    for file_path in directory.rglob("*.md"):
        if file_path.is_file():
            yield file_path


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
    Process a single markdown document with error handling.

    Parameters
    ----------
    path : Path
    docs_root : Path
    splitter : RecursiveCharacterTextSplitter

    Returns
    -------
    list[dict]
    """

    try:
        # Try UTF-8 first, then fallback to other encodings
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            logger.warning(f"UTF-8 failed for {path}, trying latin-1")
            text = path.read_text(encoding="latin-1")

        if not text.strip():
            logger.warning(f"Empty file: {path}")
            return []

        cleaned = clean_markdown(text)

        if not cleaned.strip():
            logger.warning(f"File became empty after cleaning: {path}")
            return []

        sections = split_by_headers(cleaned)

        chunks: List[Dict] = []

        for headers, section_text in sections:
            if not section_text.strip():
                continue

            split_chunks = splitter.split_text(section_text)

            for chunk in split_chunks:
                chunk = chunk.strip()
                if not chunk:
                    continue

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

    except Exception as e:
        logger.error(f"Error processing file {path}: {e}")
        return []


def process_docs(
    docs_dir: str,
    output_path: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> None:
    """
    Full preprocessing pipeline with validation and error handling.

    Parameters
    ----------
    docs_dir : str
        Directory containing markdown docs
    output_path : str
        JSONL output file
    chunk_size : int
    chunk_overlap : int
    """

    # Validate inputs
    if not docs_dir or not output_path:
        raise ValueError("docs_dir and output_path must be provided")

    if chunk_size <= 0 or chunk_overlap < 0:
        raise ValueError("chunk_size must be > 0 and chunk_overlap must be >= 0")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size")

    docs_root = Path(docs_dir)
    output_file = Path(output_path)

    logger.info(f"Processing documents from: {docs_root}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Chunk size: {chunk_size}, overlap: {chunk_overlap}")

    splitter = create_text_splitter(chunk_size, chunk_overlap)

    all_chunks: List[Dict] = []
    processed_files = 0
    total_chunks = 0

    for md_file in scan_markdown_files(docs_root):
        logger.debug(f"Processing: {md_file}")

        chunks = process_markdown_file(
            md_file,
            docs_root,
            splitter,
        )

        if chunks:
            all_chunks.extend(chunks)
            processed_files += 1
            total_chunks += len(chunks)

    # Create output directory
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create output directory: {e}")
        raise

    # Write chunks to file
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            for chunk in all_chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.error(f"Failed to write output file: {e}")
        raise

    logger.info(f"Processed {processed_files} files")
    logger.info(f"Generated {len(all_chunks)} chunks -> {output_file}")
    print(f"✅ Processed {processed_files} files, created {len(all_chunks)} chunks -> {output_file}")


if __name__ == "__main__":
    import os

    # Use environment variables or defaults
    DOCS_DIR = os.getenv("DOCS_DIR", r"/fastapi_doc/en/docs")
    OUTPUT_FILE = os.getenv("OUTPUT_FILE", "data/processed/chunks_en.jsonl")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

    try:
        process_docs(
            docs_dir=DOCS_DIR,
            output_path=OUTPUT_FILE,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"❌ Error: {e}")
        exit(1)

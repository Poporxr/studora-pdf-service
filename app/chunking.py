import hashlib
import re
from uuid import uuid4

from app.config import Settings
from app.schemas import ProcessedChunk
from app.structure import SectionText


class ApproximateEncoding:
    def encode(self, text: str) -> list[str]:
        return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

    def decode(self, tokens: list[str]) -> str:
        text = " ".join(tokens)
        return re.sub(r"\s+([.,;:!?)])", r"\1", text)


def get_encoding():
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return ApproximateEncoding()


def split_paragraphs(text: str) -> list[str]:
    normalized = re.sub(r"\n{3,}", "\n\n", text).strip()
    paragraphs = [paragraph.strip() for paragraph in normalized.split("\n\n")]

    if len(paragraphs) == 1:
        paragraphs = [
            paragraph.strip()
            for paragraph in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", normalized)
            if paragraph.strip()
        ]

    return paragraphs


def split_sentences(text: str) -> list[str]:
    # Split by sentence boundaries carefully avoiding abbreviations if possible
    # For simplicity, split on . ! ? followed by space and uppercase
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', text)
    return [s.strip() for s in sentences if s.strip()]

def build_preview(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()[:240]

def compute_chunk_hash(text: str, heading_path: list[str]) -> str:
    content = f"{' > '.join(heading_path)}\n\n{text}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def make_chunk(
    *,
    chunk_index: int,
    encoding,
    page_end: int | None,
    page_start: int | None,
    section: SectionText,
    text: str,
) -> ProcessedChunk:
    metadata = {
        "headingPath": section.section.heading_path,
        "chapterTitle": section.section.heading_path[0] if section.section.heading_path else None,
        "sectionTitle": section.section.title,
        "sectionLevel": section.section.level,
        "source": "semantic-section",
    }

    return ProcessedChunk(
        id=str(uuid4()),
        chunkIndex=chunk_index,
        sectionId=section.section.id,
        pageStart=page_start,
        pageEnd=page_end,
        text=text.strip(),
        tokenCount=len(encoding.encode(text)),
        contentPreview=build_preview(text),
        chunkHash=compute_chunk_hash(text.strip(), section.section.heading_path),
        heading=section.section.heading_path[-1] if section.section.heading_path else None,
        headingPath=section.section.heading_path,
        metadata=metadata,
    )


def chunk_sections(
    sections: list[SectionText],
    settings: Settings,
    *,
    previous_chunk_id: str | None = None,
    start_index: int = 0,
) -> list[ProcessedChunk]:
    encoding = get_encoding()
    chunks: list[ProcessedChunk] = []

    for section in sections:
        text = "\n".join(line.text for line in section.lines).strip()
        if not text:
            continue

        header = " > ".join(section.section.heading_path)
        paragraphs = split_paragraphs(text)
        current_parts: list[str] = [header]
        current_tokens = encoding.encode(header)
        current_pages = [line.page_number for line in section.lines]
        page_start = min(current_pages) if current_pages else section.section.page_start
        page_end = max(current_pages) if current_pages else section.section.page_end

        for paragraph in paragraphs:
            paragraph_tokens = encoding.encode(paragraph)

            if (
                len(current_tokens) > len(encoding.encode(header))
                and len(current_tokens) + len(paragraph_tokens) > settings.max_chunk_tokens
            ):
                chunks.append(
                    make_chunk(
                        chunk_index=start_index + len(chunks),
                        encoding=encoding,
                        page_end=page_end,
                        page_start=page_start,
                        section=section,
                        text="\n\n".join(current_parts),
                    )
                )

                overlap_tokens = (
                    current_tokens[-settings.chunk_overlap_tokens :]
                    if settings.chunk_overlap_tokens > 0
                    else []
                )
                overlap_text = encoding.decode(overlap_tokens).strip()
                current_parts = [header]
                current_tokens = encoding.encode(header)

                if overlap_text and overlap_text != header:
                    current_parts.append(overlap_text)
                    current_tokens.extend(overlap_tokens)

            if len(paragraph_tokens) > settings.max_chunk_tokens:
                # Paragraph is too big, split by sentences instead of blind slicing
                sentences = split_sentences(paragraph)
                current_sent_parts = [header]
                current_sent_tokens = encoding.encode(header)
                
                for sentence in sentences:
                    sentence_tokens = encoding.encode(sentence)
                    if len(current_sent_tokens) + len(sentence_tokens) > settings.max_chunk_tokens and len(current_sent_parts) > 1:
                        chunks.append(
                            make_chunk(
                                chunk_index=start_index + len(chunks),
                                encoding=encoding,
                                page_end=page_end,
                                page_start=page_start,
                                section=section,
                                text="\n\n".join(current_sent_parts),
                            )
                        )
                        overlap_tokens = (
                            current_sent_tokens[-settings.chunk_overlap_tokens :]
                            if settings.chunk_overlap_tokens > 0
                            else []
                        )
                        overlap_text = encoding.decode(overlap_tokens).strip()
                        current_sent_parts = [header]
                        current_sent_tokens = encoding.encode(header)
                        if overlap_text and overlap_text != header:
                            current_sent_parts.append(overlap_text)
                            current_sent_tokens.extend(overlap_tokens)
                    
                    # If a single sentence is STILL too big (rare), we fall back to blind slicing it
                    if len(sentence_tokens) > settings.max_chunk_tokens:
                        remaining = sentence_tokens
                        while remaining:
                            slice_tokens = remaining[: settings.max_chunk_tokens]
                            slice_text = encoding.decode(slice_tokens).strip()
                            if slice_text:
                                chunks.append(
                                    make_chunk(
                                        chunk_index=start_index + len(chunks),
                                        encoding=encoding,
                                        page_end=page_end,
                                        page_start=page_start,
                                        section=section,
                                        text=f"{header}\n\n{slice_text}",
                                    )
                                )
                            if len(remaining) <= settings.max_chunk_tokens:
                                break
                            overlap = (
                                slice_tokens[-settings.chunk_overlap_tokens :]
                                if settings.chunk_overlap_tokens > 0
                                else []
                            )
                            remaining = overlap + remaining[settings.max_chunk_tokens :]
                    else:
                        current_sent_parts.append(sentence)
                        current_sent_tokens.extend(sentence_tokens)
                
                if len(current_sent_parts) > 1:
                    chunks.append(
                        make_chunk(
                            chunk_index=start_index + len(chunks),
                            encoding=encoding,
                            page_end=page_end,
                            page_start=page_start,
                            section=section,
                            text="\n\n".join(current_sent_parts),
                        )
                    )
                
                current_parts = [header]
                current_tokens = encoding.encode(header)
                continue

            current_parts.append(paragraph)
            current_tokens.extend(paragraph_tokens)

        if len(current_parts) > 1:
            chunks.append(
                make_chunk(
                    chunk_index=start_index + len(chunks),
                    encoding=encoding,
                    page_end=page_end,
                    page_start=page_start,
                    section=section,
                    text="\n\n".join(current_parts),
                )
            )

    for index, chunk in enumerate(chunks):
        if index > 0:
            chunk.previous_chunk_id = chunks[index - 1].id
        elif previous_chunk_id:
            chunk.previous_chunk_id = previous_chunk_id
        if index < len(chunks) - 1:
            chunk.next_chunk_id = chunks[index + 1].id

    return chunks

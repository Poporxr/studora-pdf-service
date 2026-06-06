import re
from collections import defaultdict
from uuid import uuid4

from app.chunking import get_encoding
from app.schemas import ProcessedChunk, ProcessedSummary
from app.structure import SectionText


SUMMARY_SOURCE_VERSION = "extractive-v1"


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> list[str]:
    normalized = clean_text(text)
    if not normalized:
        return []

    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", normalized)
    if len(sentences) <= 1:
        sentences = re.split(r"\s*[•\-\u2022]\s+", normalized)

    return [
        sentence.strip()
        for sentence in sentences
        if 35 <= len(sentence.strip()) <= 320
    ]


def score_sentence(sentence: str, title: str | None = None) -> int:
    normalized = sentence.lower()
    title_terms = set()
    if title:
        title_terms = {
            term
            for term in re.sub(r"[^a-z0-9\s]", " ", title.lower()).split()
            if len(term) > 3
        }

    score = 0
    score += min(len(sentence) // 80, 3)
    score += sum(2 for term in title_terms if term in normalized)

    if re.search(r"\b(defines?|explains?|describes?|used for|important|formula|method|process|steps?)\b", normalized):
        score += 3

    if re.search(r"\b(example|therefore|however|because|advantage|disadvantage)\b", normalized):
        score += 1

    if re.search(r"\.{4,}|\bpage\s+\d+\b", normalized):
        score -= 4

    return score


def pick_key_sentences(text: str, title: str | None = None, limit: int = 5) -> list[str]:
    sentences = split_sentences(text)
    if not sentences:
        fallback = clean_text(text)
        return [fallback[:280]] if fallback else []

    ranked = sorted(
        enumerate(sentences),
        key=lambda item: (-score_sentence(item[1], title), item[0]),
    )
    selected_indexes = sorted(index for index, _ in ranked[:limit])
    selected: list[str] = []
    seen = set()

    for index in selected_indexes:
        sentence = clean_text(sentences[index])
        key = sentence.lower()
        if key in seen:
            continue
        selected.append(sentence)
        seen.add(key)

    return selected


def count_tokens(text: str) -> int:
    return len(get_encoding().encode(text))


def make_summary_text(sentences: list[str], max_chars: int = 900) -> str:
    summary = " ".join(sentences).strip()
    if len(summary) <= max_chars:
        return summary

    return summary[:max_chars].rsplit(" ", 1)[0].strip()


def build_section_text(section: SectionText, chunks: list[ProcessedChunk]) -> str:
    section_chunks = [chunk.text for chunk in chunks if chunk.section_id == section.section.id]
    if section_chunks:
        return "\n\n".join(section_chunks)

    return "\n".join(line.text for line in section.lines)


def build_summaries(
    sections: list[SectionText],
    chunks: list[ProcessedChunk],
    *,
    max_section_summaries: int = 40,
) -> list[ProcessedSummary]:
    summaries: list[ProcessedSummary] = []
    chunks_by_section: dict[str | None, list[ProcessedChunk]] = defaultdict(list)

    for chunk in chunks:
        chunks_by_section[chunk.section_id].append(chunk)

    document_parts: list[str] = []
    section_summary_count = 0

    for section in sections:
        text = build_section_text(section, chunks_by_section.get(section.section.id, []))
        cleaned = clean_text(text)
        if not cleaned:
            continue

        sentences = pick_key_sentences(cleaned, section.section.title, limit=4)
        section_summary = make_summary_text(sentences, max_chars=700)
        if section_summary:
            document_parts.append(f"{section.section.title}: {section_summary}")

        if (
            section_summary
            and section_summary_count < max_section_summaries
            and len(cleaned) >= 450
        ):
            summaries.append(
                ProcessedSummary(
                    id=str(uuid4()),
                    sectionId=section.section.id,
                    kind="SECTION",
                    title=section.section.title,
                    summary=section_summary,
                    keyPoints=sentences[:5],
                    tokenCount=count_tokens(section_summary),
                    model=None,
                    sourceVersion=SUMMARY_SOURCE_VERSION,
                )
            )
            section_summary_count += 1

    document_text = "\n".join(document_parts)
    document_sentences = pick_key_sentences(document_text, "Document overview", limit=8)
    document_summary = make_summary_text(document_sentences, max_chars=1200)

    if document_summary:
        summaries.insert(
            0,
            ProcessedSummary(
                id=str(uuid4()),
                sectionId=None,
                kind="DOCUMENT",
                title="Document overview",
                summary=document_summary,
                keyPoints=document_sentences[:8],
                tokenCount=count_tokens(document_summary),
                model=None,
                sourceVersion=SUMMARY_SOURCE_VERSION,
            ),
        )

    return summaries

import re
from dataclasses import dataclass, field
from statistics import median
from uuid import uuid4

from app.extraction import ExtractedLine, ExtractedPage
from app.schemas import ProcessedSection


@dataclass
class SectionText:
    section: ProcessedSection
    lines: list[ExtractedLine] = field(default_factory=list)


HEADING_PREFIX_PATTERN = re.compile(
    r"^(chapter|unit|module|part)\s+([0-9ivxlcdm]+|one|two|three|four|five|six|seven|eight|nine|ten)\b",
    re.IGNORECASE,
)
NUMBERED_HEADING_PATTERN = re.compile(r"^(\d+(?:\.\d+){0,4})[\.)]?\s+(.{3,120})$")
QUESTION_PATTERN = re.compile(
    r"^(\d+[\.)]|\([a-z]\)|[a-z][\.)])\s*(explain|discuss|define|describe|state|calculate|compare|differentiate|analyze|evaluate|what|why|how)\b",
    re.IGNORECASE,
)
TOC_ENTRY_PATTERN = re.compile(
    r"^(.{3,120}?)(\.{3,}|\s{3,}|\s+\|\s+).{0,30}\b\d{1,4}\s*$"
)


def is_upper_heading(text: str) -> bool:
    letters = [character for character in text if character.isalpha()]
    if len(letters) < 4:
        return False

    uppercase = sum(1 for character in letters if character.isupper())
    return uppercase / len(letters) >= 0.75


def looks_title_case(text: str) -> bool:
    words = [word for word in re.split(r"\s+", text) if word]
    if not 2 <= len(words) <= 12:
        return False

    title_words = 0
    for word in words:
        cleaned = re.sub(r"[^A-Za-z]", "", word)
        if cleaned and cleaned[0].isupper():
            title_words += 1

    return title_words / len(words) >= 0.65


def is_toc_entry(text: str) -> bool:
    normalized = text.strip()
    if not normalized:
        return False

    if TOC_ENTRY_PATTERN.match(normalized):
        return True

    if re.match(r"^\d+\s*\|\s+.{3,120}$", normalized):
        return True

    return False


def get_font_baseline(pages: list[ExtractedPage]) -> float | None:
    sizes = [
        line.font_size
        for page in pages
        for line in page.lines
        if line.font_size and line.text
    ]

    return median(sizes) if sizes else None


def score_heading(line: ExtractedLine, baseline_font_size: float | None) -> tuple[int, int]:
    text = line.text.strip()
    if not text or len(text) > 140:
        return 0, 0

    if is_toc_entry(text):
        return 0, 0

    score = 0
    level = 3

    if HEADING_PREFIX_PATTERN.match(text):
        score += 5
        level = 1

    numbered = NUMBERED_HEADING_PATTERN.match(text)
    if numbered:
        score += 4
        level = min(numbered.group(1).count(".") + 1, 4)

    if line.font_size and baseline_font_size and line.font_size >= baseline_font_size + 1.5:
        score += 3
        if line.font_size >= baseline_font_size + 4:
            level = min(level, 1)

    if line.is_bold:
        score += 2

    if is_upper_heading(text):
        score += 2
        level = min(level, 2)

    if looks_title_case(text):
        score += 1

    if text.endswith(".") and not numbered:
        score -= 2

    if QUESTION_PATTERN.match(text):
        score -= 3

    return score, level


def build_heading_path(stack: list[ProcessedSection], title: str, level: int) -> list[str]:
    parents = [section.title for section in stack if section.level < level]
    return [*parents, title]


def build_fallback_sections(pages: list[ExtractedPage]) -> list[SectionText]:
    sections: list[SectionText] = []

    for page in pages:
        if not page.text:
            continue

        lines = page.lines
        if not lines:
            lines = [
                ExtractedLine(
                    bbox=None,
                    font_name=None,
                    font_size=None,
                    is_bold=False,
                    page_number=page.page_number,
                    text=line.strip(),
                )
                for line in page.text.splitlines()
                if line.strip()
            ]

        section = ProcessedSection(
            id=str(uuid4()),
            parentSectionId=None,
            level=1,
            title=f"Page {page.page_number}",
            headingPath=[f"Page {page.page_number}"],
            pageStart=page.page_number,
            pageEnd=page.page_number,
            sortOrder=len(sections),
            confidence=0.35,
            metadata={"generatedFallback": True},
        )
        sections.append(SectionText(section=section, lines=lines))

    return sections


def detect_sections(pages: list[ExtractedPage]) -> tuple[list[SectionText], list[str]]:
    warnings: list[str] = []
    baseline_font_size = get_font_baseline(pages)
    sections: list[SectionText] = []
    stack: list[ProcessedSection] = []
    current: SectionText | None = None

    for page in pages:
        for line in page.lines:
            text = line.text.strip()
            if not text:
                continue

            if is_toc_entry(text):
                continue

            score, level = score_heading(line, baseline_font_size)
            is_heading = score >= 4

            if is_heading:
                while stack and stack[-1].level >= level:
                    stack.pop()

                section = ProcessedSection(
                    id=str(uuid4()),
                    parentSectionId=stack[-1].id if stack else None,
                    level=level,
                    title=text[:180],
                    headingPath=build_heading_path(stack, text[:180], level),
                    pageStart=page.page_number,
                    pageEnd=page.page_number,
                    sortOrder=len(sections),
                    confidence=min(score / 8, 0.98),
                    metadata={
                        "fontSize": line.font_size,
                        "isBold": line.is_bold,
                        "detectorScore": score,
                    },
                )
                current = SectionText(section=section)
                sections.append(current)
                stack.append(section)
                continue

            if current is None:
                section = ProcessedSection(
                    id=str(uuid4()),
                    parentSectionId=None,
                    level=1,
                    title="Document start",
                    headingPath=["Document start"],
                    pageStart=page.page_number,
                    pageEnd=page.page_number,
                    sortOrder=0,
                    confidence=0.4,
                    metadata={"generatedFallback": True},
                )
                current = SectionText(section=section)
                sections.append(current)
                stack = [section]

            current.lines.append(line)
            current.section.page_end = page.page_number

    if not sections:
        warnings.append("No readable text sections could be detected.")
        fallback_sections = build_fallback_sections(pages)
        if fallback_sections:
            warnings.append("Using page fallback sections.")
            return fallback_sections, warnings
        return [], warnings

    content_sections = [section for section in sections if section.lines]
    if len(content_sections) <= 1 and sum(len(page.text) for page in pages) > 2000:
        warnings.append("Document structure confidence is low; using page fallback sections.")
        return build_fallback_sections(pages), warnings

    if len(content_sections) < max(2, len(pages) // 20):
        warnings.append("Few headings were detected; section metadata may be incomplete.")

    return sections, warnings


def build_outline(sections: list[SectionText], limit: int = 80) -> list[dict[str, object]]:
    return [
        {
            "id": item.section.id,
            "level": item.section.level,
            "title": item.section.title,
            "headingPath": item.section.heading_path,
            "pageStart": item.section.page_start,
            "pageEnd": item.section.page_end,
            "confidence": item.section.confidence,
        }
        for item in sections[:limit]
    ]

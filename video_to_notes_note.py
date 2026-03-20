from __future__ import annotations

import json
from pathlib import PurePosixPath
import re
from typing import Any

from video_to_notes_schema import format_seconds


def _normalize_course_title(title: str) -> str:
    normalized = re.sub(r"^\d+(?:[.\-_]\d+)*", "", title).strip("-_. ")
    normalized = normalized or title
    return normalized


def _compact_timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    if hours:
        return f"[{hours:02d}:{minutes:02d}:{secs:02d}]"
    return f"[{minutes:02d}:{secs:02d}]"


def _note_relative_asset_path(path: str) -> str:
    normalized = " ".join(str(path).split()).strip()
    if not normalized:
        return ""
    if re.match(r"^(?:[a-z]+:|/)", normalized):
        return normalized
    posix_path = PurePosixPath(normalized)
    if posix_path.parts[:1] == ("pipeline",) and len(posix_path.parts) > 1:
        return PurePosixPath("..").joinpath(*posix_path.parts[1:]).as_posix()
    return posix_path.as_posix()


def _normalize_markdown_asset_links(markdown: str) -> str:
    pattern = re.compile(r"(\]\()([^)]+)(\))")

    def replace(match: re.Match[str]) -> str:
        prefix, raw_path, suffix = match.groups()
        normalized = _note_relative_asset_path(raw_path)
        return f"{prefix}{normalized}{suffix}"

    return pattern.sub(replace, markdown)


def _normalize_heading_text(text: str) -> str:
    normalized = " ".join(text.split()).strip()
    normalized = re.sub(
        r"^(?:本段|本节|这一段|该段|这里|接下来|随后|最后)(?:主要|重点|先|继续)?",
        "",
        normalized,
    )
    normalized = re.sub(
        r"^(?:(?:主要|重点|先|继续)?(?:讲解|介绍|说明|讨论|围绕|展开|分析|讲))(?:了|的是)?",
        "",
        normalized,
    )
    normalized = normalized.lstrip("了把将关于对从与和：:，,。 ")
    return normalized.strip()


def _split_learning_sentences(text: str, *, limit: int = 4) -> list[str]:
    parts = re.split(r"[。！？；\n]+", text)
    results = []
    for part in parts:
        normalized = " ".join(part.split()).strip(" ,，、")
        if len(normalized) >= 6:
            results.append(normalized)
        if len(results) >= limit:
            break
    return results


def _append_unique(items: list[str], candidate: str, *, limit: int) -> None:
    normalized = " ".join(candidate.split()).strip(" ,，、")
    if not normalized:
        return
    collapsed = re.sub(r"\s+", "", normalized)
    for existing in items:
        existing_collapsed = re.sub(r"\s+", "", existing)
        if collapsed == existing_collapsed or collapsed in existing_collapsed or existing_collapsed in collapsed:
            return
    if len(items) < limit:
        items.append(normalized)


def _difficulty_level(
    summary: str,
    key_points: list[str],
    corrections: list[str],
    *,
    formula_count: int = 0,
    pitfall_count: int = 0,
) -> str:
    content_length = len(" ".join([summary, *key_points]).strip())
    if formula_count > 0:
        return "高"
    if pitfall_count >= 2 or len(corrections) >= 2 or len(key_points) >= 4 or content_length >= 120:
        return "中"
    return "低"


def _build_key_points(*, summary: str, explanation_points: list[str], limit: int = 5) -> list[str]:
    key_points: list[str] = []
    if summary:
        for sentence in _split_learning_sentences(summary, limit=2):
            _append_unique(key_points, sentence, limit=limit)
    for point in explanation_points:
        _append_unique(key_points, point, limit=limit)
    return key_points


def _strip_existing_label(text: str) -> str:
    normalized = " ".join(text.split()).strip(" ,，、")
    if "：" in normalized:
        left, right = normalized.split("：", 1)
        if 1 <= len(left.strip()) <= 8 and right.strip():
            return right.strip()
    if ":" in normalized:
        left, right = normalized.split(":", 1)
        if 1 <= len(left.strip()) <= 8 and right.strip():
            return right.strip()
    return normalized


def _build_fact_points(
    *,
    definitions: list[str],
    key_points: list[str],
) -> list[str]:
    fact_points: list[str] = []
    candidate_points = [*definitions, *key_points]
    for point in candidate_points:
        normalized = _strip_existing_label(point)
        if not normalized:
            continue
        _append_unique(fact_points, normalized, limit=5)
    return fact_points


def _derive_block_kind(
    *,
    formula_candidates: list[dict[str, str]],
    visuals: list[dict[str, Any]],
    key_points: list[str],
) -> str:
    if formula_candidates:
        return "formula"
    if visuals:
        return "illustrated"
    if key_points:
        return "summary"
    return "summary"


def _build_knowledge_units(
    *,
    definitions: list[str],
    fact_points: list[str],
    formula_candidates: list[dict[str, str]],
) -> list[dict[str, Any]]:
    knowledge_units: list[dict[str, Any]] = []
    if definitions:
        knowledge_units.append({"kind": "definition", "title": "核心定义", "points": definitions[:2]})
    if fact_points:
        knowledge_units.append({"kind": "key_points", "title": "关键要点", "points": fact_points[:4]})
    if formula_candidates:
        knowledge_units.append(
            {
                "kind": "formula",
                "title": "公式与板书",
                "points": [str(item.get("latex", "")).strip() for item in formula_candidates if str(item.get("latex", "")).strip()],
            }
        )
    return knowledge_units


def _normalize_slide_title(text: str) -> str:
    normalized = " ".join(text.split()).strip().strip("-_.:：")
    if len(normalized) < 4:
        return ""
    lowered = normalized.lower()
    if lowered in {"title", "agenda", "contents"}:
        return ""
    if re.fullmatch(r"slide\s+\d+", lowered):
        return ""
    if re.fullmatch(r"第?\d+页", normalized):
        return ""
    return normalized


def _suggest_textbook_title(title: str, summary: str) -> str:
    def normalize_ascii_terms(text: str) -> str:
        return re.sub(
            r"\b[a-zA-Z]{2,12}\b",
            lambda match: match.group(0).upper() if match.group(0).lower() != match.group(0) else match.group(0),
            text,
        )

    def is_placeholder(text: str) -> bool:
        lowered = text.lower().strip()
        if not lowered:
            return True
        return bool(
            re.fullmatch(r"(?:slide|part|section|chapter|page)\s*[a-z0-9_-]*", lowered)
            or re.fullmatch(r"(?:part|section)\s*[ivx0-9]+", lowered)
        )

    def normalize_title(text: str) -> str:
        normalized = normalize_ascii_terms(" ".join(text.split()).strip())
        normalized = normalized.strip(" ，、：:;；-_.")
        return normalized

    raw_title = " ".join(title.split()).strip()
    summary_text = " ".join(summary.split()).strip()
    normalized_title = normalize_title(raw_title)
    if normalized_title and not is_placeholder(normalized_title):
        return normalized_title

    normalized_summary = normalize_title(_normalize_heading_text(summary_text))
    if normalized_summary:
        normalized_summary = re.split(
            r"[。；：]|，(?:并|以及|随后|接着|然后|同时|并且|用|说明|解释|强调|指出|结合|通过|举|补充)",
            normalized_summary,
            maxsplit=1,
        )[0].strip(" ，、：:;；-_.")
    if normalized_summary:
        return normalized_summary

    fallback = normalize_ascii_terms(raw_title or summary_text).strip()
    return fallback or "课程小节"


def sanitize_note_body_timestamps(markdown: str) -> str:
    lines = markdown.splitlines()
    in_lecture = False
    in_summary_table = False
    sanitized: list[str] = []
    timestamp_pattern = re.compile(r"\s*\[\d{2}:\d{2}(?::\d{2})?\](?=(?:\s|$))")

    for line in lines:
        stripped = line.strip()
        if stripped == "## 知识小结":
            in_summary_table = True
        elif stripped.startswith("## ") and stripped != "## 知识小结":
            in_summary_table = False

        if stripped == "## 课程讲义":
            in_lecture = True
            sanitized.append(line)
            continue
        if stripped == "## 关键截图索引":
            in_lecture = False
            sanitized.append(line)
            continue

        keep_line = (
            not in_lecture
            or in_summary_table
            or stripped.startswith("#")
            or re.match(r"^\s*-\s+[一二三四五六七八九十]+、", line)
            or re.match(r"^\s*-\s+\d+\.\s", line)
            or re.match(r"^\s*!\[", line)
            or "|" in line
        )
        if keep_line:
            sanitized.append(line)
            continue

        sanitized.append(timestamp_pattern.sub("", line))

    return _normalize_markdown_asset_links("\n".join(sanitized))


def render_ppt_alignment_debug_markdown(*, title: str, visual_alignment: dict[str, Any]) -> str:
    lines = [
        f"# PPT Alignment Debug: {title}",
        "",
        f"- visual_source_mode: `{str(visual_alignment.get('visual_source_mode', ''))}`",
        f"- segment_count: `{int(visual_alignment.get('segment_count', 0) or 0)}`",
        f"- ppt_slide_count: `{int(visual_alignment.get('ppt_slide_count', 0) or 0)}`",
        "",
    ]

    for segment in visual_alignment.get("segments", []):
        segment_id = str(segment.get("segment_id", "")).strip()
        label = str(segment.get("label", "")).strip()
        summary = str(segment.get("summary", "")).strip()
        selection_mode = str(segment.get("selection_mode", "")).strip()
        lines.append(f"## {segment_id} {label}".strip())
        lines.append("")
        if summary:
            lines.append(f"- summary: {summary}")
        lines.append(f"- selection_mode: `{selection_mode}`")
        if segment.get("reject_reason"):
            lines.append(f"- reject_reason: {str(segment.get('reject_reason', '')).strip()}")
        if segment.get("borrow_reason"):
            lines.append(f"- borrow_reason: {str(segment.get('borrow_reason', '')).strip()}")

        trace = segment.get("ppt_sequence_trace", {})
        if isinstance(trace, dict) and trace:
            lines.append("- selected_path_trace:")
            lines.append(f"  - slide_index: `{trace.get('selected_slide_index')}`")
            lines.append(f"  - base_score: `{trace.get('base_score')}`")
            lines.append(f"  - sequence_step_score: `{trace.get('sequence_step_score')}`")
            lines.append(f"  - cumulative_score: `{trace.get('cumulative_score')}`")
            lines.append(f"  - global_path_score: `{trace.get('global_path_score')}`")
            lines.append(f"  - previous_slide_index: `{trace.get('previous_slide_index')}`")
            lines.append(f"  - sequence_reason: `{trace.get('sequence_reason')}`")

        selected_visuals = segment.get("selected_visuals", [])
        if isinstance(selected_visuals, list) and selected_visuals:
            lines.append("- selected_visuals:")
            for visual in selected_visuals[:2]:
                lines.append(
                    f"  - slide `{visual.get('slide_index', '')}` | base=`{visual.get('base_score', visual.get('score', ''))}` | "
                    f"seq=`{visual.get('sequence_score', '')}` | cum=`{visual.get('cumulative_score', '')}` | "
                    f"global=`{visual.get('global_path_score', '')}` | "
                    f"reason=`{visual.get('sequence_reason', '')}` | path=`{visual.get('relative_path', '')}`"
                )

        candidate_visuals = segment.get("candidate_visuals", [])
        ppt_candidates = [
            candidate
            for candidate in candidate_visuals
            if isinstance(candidate, dict) and str(candidate.get("source", "")).strip() == "ppt_slide"
        ]
        if ppt_candidates:
            lines.append("- candidate_ppt_slides:")
            for candidate in ppt_candidates[:5]:
                rejection = str(candidate.get("sequence_rejection_reason", "")).strip()
                gap = candidate.get("sequence_score_gap", "")
                lines.append(
                    f"  - slide `{candidate.get('slide_index', '')}` | selected=`{candidate.get('sequence_selected', False)}` | "
                    f"base=`{candidate.get('base_score', candidate.get('score', ''))}` | seq=`{candidate.get('sequence_score', '')}` | "
                    f"cum=`{candidate.get('cumulative_score', '')}` | global=`{candidate.get('global_path_score', '')}` | prev=`{candidate.get('sequence_previous_slide_index', '')}` | "
                    f"reason=`{candidate.get('sequence_reason', '')}` | reject=`{rejection}` | gap=`{gap}`"
                )

        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _extract_formula_candidates(*, summary: str, explanation_points: list[str], visuals: list[dict[str, Any]]) -> list[dict[str, str]]:
    formulas: list[dict[str, str]] = []
    seen: set[str] = set()
    sources = [
        summary,
        *explanation_points,
        *[str(item.get("ocr_text", "")).strip() for item in visuals if str(item.get("ocr_text", "")).strip()],
    ]

    def looks_like_formula(text: str) -> bool:
        candidate = text.strip()
        if not candidate:
            return False
        if "\\frac" in candidate or "\\sum" in candidate or "\\log" in candidate:
            return True
        math_chars = sum(1 for char in candidate if char in "=+-*/^_()[]{}<>∑Σ√≤≥≈")
        alpha_num = sum(1 for char in candidate if char.isalnum())
        return math_chars >= 2 and alpha_num >= 3

    def extract_formula_fragment(text: str) -> str:
        candidate = " ".join(text.split()).strip()
        match = re.search(r"([A-Za-z][A-Za-z0-9_(){}\[\]]*\s*=\s*[^，。；;\n]+)", candidate)
        if match:
            return match.group(1).strip()
        return candidate

    for source in sources:
        for chunk in re.split(r"[；;\n]+", source):
            candidate = extract_formula_fragment(chunk)
            if not looks_like_formula(candidate):
                continue
            if candidate in seen:
                continue
            seen.add(candidate)
            formulas.append(
                {
                    "name": f"expression_{len(formulas) + 1}",
                    "latex": candidate,
                    "evidence": candidate,
                }
            )
            if len(formulas) >= 3:
                return formulas
    return formulas


def _is_internal_process_sentence(text: str) -> bool:
    normalized = " ".join(text.split()).strip().lower()
    if not normalized:
        return False
    internal_markers = (
        "修正",
        "误听",
        "误识别",
        "ocr",
        "证据不足",
        "不补写公式",
        "未编造公式",
        "raw -> cleaned",
        "统一为",
        "理顺句子",
        "同音误识别",
        "口误",
    )
    return any(marker in normalized for marker in internal_markers)
def _reader_facing_points(points: list[str], *, limit: int = 3) -> list[str]:
    filtered: list[str] = []
    for point in points:
        normalized = " ".join(str(point).split()).strip(" ,，、")
        if not normalized or _is_internal_process_sentence(normalized):
            continue
        if len(normalized) < 8:
            continue
        _append_unique(filtered, normalized, limit=limit)
    return filtered


def _derive_section_title(summary: str, cleaned_text: str, index: int, *, slide_title: str = "") -> str:
    preferred_slide_title = _normalize_slide_title(slide_title)
    if preferred_slide_title:
        return _suggest_textbook_title(preferred_slide_title, summary or cleaned_text)
    source = _normalize_heading_text(summary or cleaned_text)
    source = re.split(
        r"[。；：]|，(?:并|以及|随后|接着|然后|同时|并且|用|说明|解释|强调|指出|结合|通过|举|补充)",
        source,
        maxsplit=1,
    )[0].strip()
    source = source.strip(" -")
    if len(source) < 6 and cleaned_text:
        fallback_sentences = _split_learning_sentences(cleaned_text, limit=1)
        if fallback_sentences:
            source = _normalize_heading_text(fallback_sentences[0])
    if len(source) > 30:
        source = source[:30].rstrip("与和的里中")
    source = source.rstrip(" ，、")
    return _suggest_textbook_title(source or f"课程小节 {index}", summary or cleaned_text)


def build_note_outline(*, title: str, visual_alignment: dict[str, Any] | None = None) -> dict[str, Any]:
    visual_alignment = visual_alignment or {}
    segments = visual_alignment.get("segments", [])
    course_title = _normalize_course_title(title)
    chapter_id = "chapter_001"
    chapter_start = float(segments[0].get("start", 0.0) or 0.0) if segments else 0.0
    chapter_end = float(segments[-1].get("end", chapter_start) or chapter_start) if segments else chapter_start
    sections = []
    previous_title = ""
    for index, segment in enumerate(segments, start=1):
        selected_visuals = segment.get("selected_visuals", [])
        primary_visual = selected_visuals[0] if isinstance(selected_visuals, list) and selected_visuals else {}
        slide_title = str(primary_visual.get("slide_title", "")).strip() if isinstance(primary_visual, dict) else ""
        title_text = _derive_section_title(
            str(segment.get("summary", "")).strip(),
            str(segment.get("cleaned_text", "")).strip() or str(segment.get("raw_text", "")).strip(),
            index,
            slide_title=slide_title,
        )
        if title_text == previous_title:
            fallback_title = _derive_section_title(
                str(segment.get("summary", "")).strip(),
                str(segment.get("cleaned_text", "")).strip() or str(segment.get("raw_text", "")).strip(),
                index,
                slide_title="",
            )
            if fallback_title and fallback_title != previous_title:
                title_text = fallback_title
            else:
                title_text = f"{title_text}（续）"
        previous_title = title_text
        sections.append(
            {
                "section_id": f"section_{index:03d}",
                "segment_id": str(segment.get("segment_id", "")).strip(),
                "index": index,
                "title": title_text,
                "start": float(segment.get("start", 0.0) or 0.0),
                "end": float(segment.get("end", 0.0) or 0.0),
                "time_label": format_seconds(float(segment.get("start", 0.0) or 0.0)),
                "source_title": slide_title or title_text,
            }
        )

    chapters = []
    if sections:
        chapters.append(
            {
                "chapter_id": chapter_id,
                "index_label": "一",
                "title": course_title,
                "start": chapter_start,
                "end": chapter_end,
                "time_label": format_seconds(chapter_start),
                "sections": sections,
            }
        )

    return {
        "title": title,
        "chapter_count": len(chapters),
        "chapters": chapters,
    }


def build_note_blocks(
    *,
    note_outline: dict[str, Any] | None = None,
    visual_alignment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    note_outline = note_outline or {}
    visual_alignment = visual_alignment or {}
    segment_map = {
        str(item.get("segment_id", "")).strip(): item
        for item in visual_alignment.get("segments", [])
    }
    blocks = []
    for chapter in note_outline.get("chapters", []):
        for section in chapter.get("sections", []):
            segment_id = str(section.get("segment_id", "")).strip()
            aligned = segment_map.get(segment_id, {})
            cleaned_text = str(aligned.get("cleaned_text", "")).strip()
            raw_text = str(aligned.get("raw_text", "")).strip()
            explanation_points = _reader_facing_points(
                _split_learning_sentences(cleaned_text or raw_text, limit=6),
                limit=6,
            )
            if not explanation_points and str(aligned.get("summary", "")).strip():
                explanation_points = _reader_facing_points([str(aligned.get("summary", "")).strip()], limit=2)
            summary_text = _normalize_heading_text(str(aligned.get("summary", "")).strip())
            definitions = _reader_facing_points([summary_text], limit=2) if summary_text else []
            formula_candidates = _extract_formula_candidates(
                summary=str(aligned.get("summary", "")).strip(),
                explanation_points=explanation_points,
                visuals=aligned.get("selected_visuals", []),
            )
            key_points = _build_key_points(summary=summary_text, explanation_points=explanation_points, limit=5)
            mechanism_points = key_points[:4]
            comparison_points: list[str] = []
            examples: list[str] = []
            pitfall_points: list[str] = []
            fact_points = _build_fact_points(
                definitions=definitions,
                key_points=key_points,
            )
            source_visuals = aligned.get("selected_visuals", [])
            block_kind = _derive_block_kind(
                formula_candidates=formula_candidates,
                visuals=source_visuals if isinstance(source_visuals, list) else [],
                key_points=key_points,
            )
            knowledge_units = _build_knowledge_units(
                definitions=definitions,
                fact_points=fact_points,
                formula_candidates=formula_candidates,
            )
            primary_visual = source_visuals[0] if isinstance(source_visuals, list) and source_visuals else {}
            blocks.append(
                {
                    "block_id": f"block_{len(blocks) + 1:03d}",
                    "chapter_id": str(chapter.get("chapter_id", "")).strip(),
                    "section_id": str(section.get("section_id", "")).strip(),
                    "segment_id": segment_id,
                    "title": str(section.get("title", "")).strip(),
                    "time_label": str(section.get("time_label", "")).strip(),
                    "start": float(aligned.get("start", section.get("start", 0.0)) or 0.0),
                    "end": float(aligned.get("end", section.get("end", 0.0)) or 0.0),
                    "summary": str(aligned.get("summary", "")).strip(),
                    "explanation_points": explanation_points,
                    "key_points": key_points[:5],
                    "fact_points": fact_points,
                    "corrections": [str(item).strip() for item in aligned.get("issues", []) if str(item).strip()][:3],
                    "concepts": [str(section.get("title", "")).strip()] if str(section.get("title", "")).strip() else [],
                    "definitions": definitions,
                    "mechanism_points": mechanism_points,
                    "comparison_points": comparison_points,
                    "examples": examples,
                    "pitfalls": pitfall_points,
                    "difficulty": _difficulty_level(
                        str(aligned.get("summary", "")).strip(),
                        key_points[:5],
                        pitfall_points,
                        formula_count=len(formula_candidates),
                        pitfall_count=len(pitfall_points),
                    ),
                    "timestamp_ref": _compact_timestamp(float(aligned.get("start", section.get("start", 0.0)) or 0.0)),
                    "formula_candidates": formula_candidates,
                    "knowledge_units": knowledge_units,
                    "block_kind": block_kind,
                    "visuals": source_visuals,
                    "primary_visual_source": str(primary_visual.get("source", "")).strip() if isinstance(primary_visual, dict) else "",
                    "primary_visual_title": str(primary_visual.get("slide_title", "")).strip() if isinstance(primary_visual, dict) else "",
                    "selection_mode": str(aligned.get("selection_mode", "")).strip(),
                }
            )
    return {
        "block_count": len(blocks),
        "blocks": blocks,
    }


def render_note_markdown(
    *,
    title: str,
    source_video: str,
    duration_seconds: float,
    transcript_excerpt: list[str],
    note_outline: dict[str, Any] | None = None,
    note_blocks: dict[str, Any] | None = None,
    corrections: list[dict[str, Any]] | None = None,
    frames: list[dict[str, Any]] | None = None,
) -> str:
    note_outline = note_outline or {}
    note_blocks = note_blocks or {}
    corrections = corrections or []
    frames = frames or []
    chapters = note_outline.get("chapters", [])
    block_map = {str(block.get("section_id", "")).strip(): block for block in note_blocks.get("blocks", [])}
    section_count = sum(len(chapter.get("sections", [])) for chapter in chapters)
    correction_count = len(
        [item for item in corrections if str(item.get("raw", "")).strip() and str(item.get("cleaned", "")).strip()]
    )
    summary_blocks = note_blocks.get("blocks", [])

    lines = [
        f"# {title} 学习讲义",
        "",
        f"> 来源视频：`{source_video}` | 视频时长：`{format_seconds(duration_seconds)}` | 小节数：`{section_count}` | 关键纠错：`{correction_count}`",
        "",
        "## 课程目录",
        "",
    ]

    if chapters:
        for chapter in chapters:
            lines.append(f"### {chapter.get('index_label', '一')}、{chapter.get('title', '')} {chapter.get('time_label', '')}".rstrip())
            lines.append("")
            for section in chapter.get("sections", []):
                block = block_map.get(str(section.get("section_id", "")).strip(), {})
                timestamp_ref = str(block.get("timestamp_ref", "")).strip() or _compact_timestamp(float(section.get("start", 0.0) or 0.0))
                lines.append(f"{section.get('index', 1)}. {section.get('title', '')} {timestamp_ref}".rstrip())
            lines.append("")
    elif transcript_excerpt:
        lines.extend([f"- {entry}" for entry in transcript_excerpt])
        lines.append("")
    else:
        lines.append("- 未生成课程目录")
        lines.append("")

    lines.extend(["## 知识小结", ""])
    if summary_blocks:
        lines.append("| 时间 | 小节 | 核心内容 | 易错点 | 难度 |")
        lines.append("| --- | --- | --- | --- | --- |")
        for block in summary_blocks:
            concept = str(block.get("concepts", [""])[0] if block.get("concepts") else block.get("title", "")).strip()
            summary = str(block.get("summary", "")).strip() or "未生成"
            pitfall = "；".join(str(item).strip() for item in block.get("pitfalls", []) if str(item).strip()) or "-"
            difficulty = str(block.get("difficulty", "")).strip() or "中"
            timestamp_ref = str(block.get("timestamp_ref", "")).strip()
            lines.append(f"| {timestamp_ref} | {concept} | {summary} | {pitfall} | {difficulty} |")
        lines.append("")
    else:
        lines.append("- 未生成知识小结")
        lines.append("")

    lines.extend(["## 课程讲义", ""])

    if chapters:
        for chapter in chapters:
            lines.append(f"### {chapter.get('index_label', '一')}、{chapter.get('title', '')} {chapter.get('time_label', '')}".rstrip())
            lines.append("")
            for section in chapter.get("sections", []):
                block = block_map.get(str(section.get("section_id", "")).strip(), {})
                timestamp_ref = str(block.get("timestamp_ref", "")).strip() or _compact_timestamp(float(section.get("start", 0.0) or 0.0))
                lines.append(f"#### {section.get('index', 1)}. {section.get('title', '')} {timestamp_ref}".rstrip())
                lines.append("")
                item_index = 1
                lines.append(f"{item_index}）核心定义卡片")
                if block.get("concepts") or str(block.get("summary", "")).strip():
                    lines.append(f"> 概念：{str(block.get('concepts', [''])[0] if block.get('concepts') else section.get('title', '')).strip()}")
                    if block.get("definitions"):
                        lines.append(f"> 定义：{str(block.get('definitions', [''])[0]).strip()}")
                    else:
                        lines.append(f"> 定义：{str(block.get('summary', '')).strip()}")
                else:
                    lines.append("- 未生成核心内容")
                lines.append("")
                item_index += 1
                lines.append(f"{item_index}）知识框架")
                points = block.get("fact_points", []) or block.get("mechanism_points", []) or block.get("key_points", []) or block.get("explanation_points", [])
                if points:
                    lines.extend([f"- {str(item).strip()}" for item in points if str(item).strip()])
                else:
                    lines.append("- 未生成知识要点")
                lines.append("")
                if block.get("comparison_points"):
                    item_index += 1
                    lines.append(f"{item_index}）关键对比")
                    lines.extend([f"- {str(item).strip()}" for item in block.get("comparison_points", []) if str(item).strip()])
                    lines.append("")
                if block.get("examples"):
                    item_index += 1
                    lines.append(f"{item_index}）例子与直觉")
                    lines.extend([f"- {str(item).strip()}" for item in block.get("examples", []) if str(item).strip()])
                    lines.append("")
                if block.get("formula_candidates"):
                    item_index += 1
                    lines.append(f"{item_index}）公式与板书")
                    for formula in block.get("formula_candidates", []):
                        lines.append(f"- {formula.get('name', 'formula')}: `${formula.get('latex', '')}`")
                    lines.append("")
                if block.get("pitfalls"):
                    item_index += 1
                    lines.append(f"{item_index}）易错点")
                    lines.extend([f"- {str(item).strip()}" for item in block.get("pitfalls", []) if str(item).strip()])
                    lines.append("")
                if block.get("explanation_points"):
                    item_index += 1
                    lines.append(f"{item_index}）讲解展开")
                    lines.extend([f"- {str(item).strip()}" for item in block.get("explanation_points", []) if str(item).strip()])
                    lines.append("")
                visuals = block.get("visuals", [])[:1]
                if visuals:
                    item_index += 1
                    lines.append(f"{item_index}）配图与板书")
                    for visual in visuals:
                        rel = _note_relative_asset_path(str(visual.get("relative_path", "")).strip())
                        if not rel:
                            continue
                        ts = format_seconds(float(visual.get("timestamp", 0.0) or 0.0))
                        lines.append("")
                        lines.append(f"![{ts}]({rel})")
                    lines.append("")
    else:
        lines.append("- 未生成详细讲义")
        lines.append("")

    lines.extend(["", "## 关键截图索引", ""])
    if frames:
        for frame in frames:
            ts = format_seconds(float(frame["timestamp"]))
            rel = _note_relative_asset_path(str(frame["relative_path"]).strip())
            if not rel:
                continue
            lines.append(f"### {ts}")
            lines.append("")
            lines.append(f"![{ts}]({rel})")
            lines.append("")
    else:
        lines.append("- 未生成关键截图")
        lines.append("")

    return "\n".join(lines)


def build_note_generation_prompt(
    *,
    title: str,
    source_video: str,
    note_outline: dict[str, Any] | None = None,
    note_blocks: dict[str, Any] | None = None,
    corrections: list[dict[str, Any]] | None = None,
) -> str:
    note_outline = note_outline or {}
    note_blocks = note_blocks or {}
    corrections = corrections or []
    chapter_lines: list[str] = []
    for chapter in note_outline.get("chapters", []):
        chapter_lines.append(
            f"- {chapter.get('index_label', '一')}、{chapter.get('title', '')} {chapter.get('time_label', '')}".rstrip()
        )
        for section in chapter.get("sections", [])[:20]:
            section_ts = _compact_timestamp(float(section.get("start", 0.0) or 0.0))
            raw_section_title = str(section.get("title", "")).strip()
            suggested_section_title = _suggest_textbook_title(raw_section_title, raw_section_title)
            chapter_lines.append(f"  - {section_ts} {suggested_section_title}".rstrip())

    block_lines: list[str] = []
    for block in note_blocks.get("blocks", [])[:20]:
        timestamp_ref = str(block.get("timestamp_ref", "")).strip()
        raw_title_text = str(block.get("title", "")).strip()
        summary = str(block.get("summary", "")).strip()
        title_text = _suggest_textbook_title(raw_title_text, summary)
        block_kind = str(block.get("block_kind", "")).strip()
        definitions = _reader_facing_points([str(item).strip() for item in block.get("definitions", []) if str(item).strip()], limit=2)
        mechanisms = _reader_facing_points([str(item).strip() for item in block.get("mechanism_points", []) if str(item).strip()], limit=3)
        fact_points = _reader_facing_points([str(item).strip() for item in block.get("fact_points", []) if str(item).strip()], limit=4)
        if not mechanisms:
            mechanisms = _reader_facing_points([str(item).strip() for item in block.get("key_points", []) if str(item).strip()], limit=3)
        comparisons = _reader_facing_points([str(item).strip() for item in block.get("comparison_points", []) if str(item).strip()], limit=2)
        examples = _reader_facing_points([str(item).strip() for item in block.get("examples", []) if str(item).strip()], limit=2)
        pitfalls = _reader_facing_points([str(item).strip() for item in block.get("pitfalls", []) if str(item).strip()], limit=2)
        explanations = _reader_facing_points([str(item).strip() for item in block.get("explanation_points", []) if str(item).strip()], limit=3)
        visuals = block.get("visuals", [])[:2]
        formulas = block.get("formula_candidates", [])[:3]

        block_lines.append(f"- {title_text}".rstrip())
        if timestamp_ref:
            block_lines.append(f"  - 小节时间：{timestamp_ref}")
        if block_kind:
            block_lines.append(f"  - 讲义类型：{block_kind}")
        if summary:
            block_lines.append(f"  - 摘要：{summary}")
        if definitions:
            block_lines.append("  - 定义：")
            block_lines.extend([f"    - {item}" for item in definitions])
        if fact_points:
            block_lines.append("  - 事实提纲：")
            block_lines.extend([f"    - {item}" for item in fact_points])
        elif mechanisms:
            block_lines.append("  - 原理机制：")
            block_lines.extend([f"    - {item}" for item in mechanisms])
        if comparisons:
            block_lines.append("  - 关键对比：")
            block_lines.extend([f"    - {item}" for item in comparisons])
        if examples:
            block_lines.append("  - 例子与直觉：")
            block_lines.extend([f"    - {item}" for item in examples])
        if pitfalls:
            block_lines.append("  - 易错点：")
            block_lines.extend([f"    - {item}" for item in pitfalls])
        elif explanations:
            block_lines.append("  - 补充线索：")
            block_lines.extend([f"    - {item}" for item in explanations[:2]])
        if formulas:
            block_lines.append("  - 公式候选：")
            for formula in formulas:
                block_lines.append(
                    f"    - {formula.get('name', 'formula')}: {formula.get('latex', '')} | OCR={formula.get('evidence', '')}"
                )
        if visuals:
            block_lines.append("  - 配图候选：")
            for visual in visuals:
                visual_ts = _compact_timestamp(float(visual.get("timestamp", 0.0) or 0.0))
                visual_path = _note_relative_asset_path(str(visual.get("relative_path", "")).strip())
                block_lines.append(f"    - {visual_ts} {visual_path}".rstrip())

    formula_lines: list[str] = []
    seen_formula_keys: set[tuple[str, str]] = set()
    for block in note_blocks.get("blocks", []):
        for formula in block.get("formula_candidates", [])[:2]:
            key = (str(formula.get("name", "")).strip(), str(formula.get("latex", "")).strip())
            if key in seen_formula_keys:
                continue
            seen_formula_keys.add(key)
            formula_lines.append(
                f"- {block.get('timestamp_ref', '')} {block.get('title', '')}: {formula.get('name', 'formula')} => {formula.get('latex', '')} | OCR={formula.get('evidence', '')}"
            )
    return "\n".join(
        [
            f"# Note Generation Prompt: {title}",
            "",
            "## 任务目标",
            "你是一位资深计算机课程讲义编辑。请把输入材料整理成适合考前复习和课后回顾的提纲式学习讲义，而不是展示生成过程或审阅痕迹。",
            "如果小节已经绑定了 PPT 页标题或 PPT 配图，请优先沿用该标题体系，让输出更像教材目录，而不是口语化摘要或课堂复述。",
            "输入里的标题只是素材线索，不是最终标题定稿；如果素材标题像口播标题、页内临时标题或不完整短语，请由你重写成教材式标题。",
            "",
            "## 输出合同",
            "",
            "1. 你必须输出多级列表，拒绝大段平铺文字；正文优先使用提纲式缩进，不要把内容写成长段讲解稿。",
            "2. 整体结构要尽量接近教材提纲：",
            "   - `一、主题`",
            "   - `1. 小节`",
            "   - `1）子点`",
            "   - `bullet` 要点",
            "3. 每个小节优先围绕知识单元组织，而不是机械套模板。建议优先出现这些内容：",
            "   - `核心定义卡片`（用最短的话界定概念）",
            "   - `知识框架`（拆成 3-6 个关键要点）",
            "   - 必要时再补 `展开讲解`、`例子/直觉`、`易错点`、`配图与公式`",
            "4. 默认每个小节先写 3-5 个高信息密度 bullet；除非确有必要，不写长段展开。",
            "5. 不要为每一节强行凑满所有栏目；如果某节不需要公式或长解释，就不要硬写。",
            "6. 小结标题要优先写成教材式标题，优先吸收 PPT 页标题，不要直接复述口播；标题尽量控制在 8-18 字。",
            "   - 避免使用临时页标题、口播式标题、问句式标题或明显截断的长标题。",
            "   - 更偏好“概念名 / 机制名 / 约定名 / 输出头 / 编码流程”这类教材式标题。",
            "7. 精确时间戳只保留在主标题层级、课程目录和知识小结表中。",
            "   - 你必须输出类似 `[02:48]` 这样的精确时间戳引用，但不要给每个 bullet 句子单独附时间。",
            "   - 小节标题可带时间，节内 `1）2）3）` 子标题和 bullet 要点默认不带时间。",
            "   - 除目录、小节标题、知识小结表和图片 Markdown alt 文本外，正文句子末尾不得追加 `[mm:ss]`。",
            "8. 需要公式时输出 LaTeX：",
            "   - 当视频讲解涉及任何数学概念（概率、极大似然、损失函数、信息熵、信息增益、Softmax、Logistic、矩阵等），只要 `formula_candidates` 或明确的 OCR 抓到了对应的英文/符号线索，**必须**输出高亮的 LaTeX 公式块（`$$...$$` 或 `$ ... $`）。",
            "   - 你必须参考 OCR 证据写出正确的 LaTeX 公式，不能凭空脑补或只写口头描述。",
            "   - 若无确凿的 OCR 或转写支撑，就不要写公式。",
            "   - 同一公式不要在多个相邻小节反复整块重复；首次完整写出，后文只在必要时引用其含义。",
            "9. 正文顶部必须输出 `## 知识小结` 及其 Markdown 表格，也就是知识点总结表格。字段固定为：`时间 | 知识点 | 核心内容 | 易错点 | 难度评级`。",
            "10. 如果某节存在图片候选，必须在对应小节的 `配图与公式` 区块里使用真实 Markdown 图片标签：`![timestamp](relative_path)`。",
            "11. 正文只保留面向读者的知识内容，不要暴露内部生成过程。",
            "   - 不要输出“术语修正”“易错转写修正”“OCR 证据不足”“不补写公式”“未编造公式”等内部说明。",
            "   - `corrections` 只用于你在幕后纠正表达，不要在正文中展示 `raw -> cleaned` 的修订痕迹。",
            "   - 如果没有可靠公式，就安静地不写公式，不要解释为什么没写。",
            "12. 严禁写课堂纪要式句子，例如“老师这里讲了什么”“这一段主要介绍了什么”“接下来我们来看”。",
            "13. 优先保留定义、组成、流程、作用、对比、易错点；压缩过程性描述和重复解释。",
            "14. `核心定义卡片`、`知识框架` 这两个标签只需在全文关键大节中出现即可；多数小节可直接写成 `1）2）3）` 的提纲子点，不必反复显式打印这两个标签。",
            "",
            "## 课程目录摘要",
            "",
            f"- source_video: `{source_video}`",
            *(chapter_lines or ["- 无章节摘要"]),
            "",
            "## 小节素材摘要",
            "",
            *(block_lines or ["- 无小节摘要"]),
            "",
            "## 公式候选",
            "",
            "- formula_candidates:",
            *(formula_lines or ["- 无显式公式候选"]),
            "",
            "## 执行指令",
            "",
            "- 用上面的课程目录摘要和小节素材摘要写最终讲义，不要照抄内部字段名。",
            "- 写作目标是“复习讲义”而不是“课程纪要”；默认先提炼结论，再补最少量解释。",
            "- 如果 `formula_candidates` 有匹配特征，必须植入对应的数学解释，绝不可漏掉。",
            "- 你必须仔细检查每一小节是否带有图片路径，切忌漏掉能帮助理解的截图。",
            "- 全文必须出现 `## 知识小结`、`核心定义卡片` 和 `知识框架` 的原文字眼，以满足自动验证；但这些字眼不必在每个小节机械重复。",
            "- 不要在每个小节前重复抄写同一章标题；一章下面直接顺着列出小节即可。",
            "- 相邻小节如果公式、配图或解释高度重复，请合并或压缩，避免同一内容反复出现。",
            "- 不要把审阅留痕、纠错过程、OCR 充分性判断写进主文档正文。",
            "- 讲义体感应像清晰、克制、适合复习的教材讲义，而不是审阅报告、调试日志或老师口播整理稿。",
            "- 上面的小节标题只是素材标题；如果标题不自然，请直接重写成教材式标题，不要照抄。",
            "- 如果正文 bullet 末尾出现 `[00:00]` 之类时间戳，视为不合格格式，必须删掉，只保留对应小节标题上的时间。",
            "",
        ]
    )

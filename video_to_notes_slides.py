from __future__ import annotations

import re
import shutil
import unicodedata
from pathlib import Path
from typing import Any

import fitz


SLIDES_PAYLOAD_VERSION = 4

def resolve_slides_path(_input_video: Path, explicit_path: str | None = None) -> Path | None:
    if explicit_path:
        candidate = Path(explicit_path).expanduser().resolve()
        return candidate if candidate.exists() else None
    return None


def _normalize_slide_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def _is_supported_slide_char(char: str) -> bool:
    if not char:
        return False
    if char.isspace():
        return True
    if char.isascii() and (char.isalnum() or char in "+-_=:/%()[]{}.,，。；：、<>×*·•|"):
        return True
    if "\u4e00" <= char <= "\u9fff":
        return True
    if char in {"α", "β", "γ", "δ", "θ", "λ", "μ", "σ", "τ", "φ", "ψ", "ω", "ℎ"}:
        return True
    return False


def _sanitize_slide_text(text: str) -> str:
    normalized = _normalize_slide_text(text)
    cleaned_tokens: list[str] = []
    for token in normalized.split():
        stripped = "".join(
            char
            for char in token
            if not unicodedata.category(char).startswith("M") and unicodedata.category(char) != "Cf"
        )
        if not stripped:
            continue
        if any(not _is_supported_slide_char(char) for char in stripped):
            allowed_ratio = sum(1 for char in stripped if _is_supported_slide_char(char)) / max(len(stripped), 1)
            if allowed_ratio < 0.85:
                continue
        if re.search(r"[ǍǓǗǝǜƲƭƹ]", stripped):
            continue
        cleaned_tokens.append(stripped)
    cleaned = _normalize_slide_text(" ".join(cleaned_tokens))
    cleaned = re.sub(r"\s+([,.;:，。；：、])", r"\1", cleaned)
    return cleaned


def _text_has_slide_noise(text: str) -> bool:
    if not text.strip():
        return False
    if re.search(r"[ǍǓǗǝǜƲƭƹ֩]", text):
        return True
    noisy_chars = sum(1 for char in text if not _is_supported_slide_char(char) and not char.isspace())
    return noisy_chars >= 2


def _derive_slide_title(text: str, index: int) -> str:
    lines = [line.strip() for line in re.split(r"[\n|·•]+", text) if line.strip()]
    for line in lines:
        if 2 <= len(line) <= 80 and not _text_has_slide_noise(line):
            tokens = line.split()
            if len(tokens) >= 4 and len(tokens[0]) >= 4:
                tail_tokens = tokens[1:]
                if tail_tokens and sum(1 for token in tail_tokens if len(token) <= 2) >= 3:
                    candidate = tokens[0]
                    if 2 <= len(candidate) <= 40:
                        return candidate
            if len(tokens) > 1:
                compact_tokens: list[str] = []
                single_char_run = 0
                for token in tokens:
                    if re.fullmatch(r"[\u4e00-\u9fffA-Za-z]", token):
                        single_char_run += 1
                    else:
                        single_char_run = 0
                    if single_char_run >= 3 and compact_tokens:
                        break
                    compact_tokens.append(token)
                candidate = " ".join(compact_tokens).strip()
            else:
                candidate = line
            if 2 <= len(candidate) <= 40:
                return candidate
    return f"Slide {index}"


def _compact_existing_title(title: str, index: int) -> str:
    normalized = _normalize_slide_text(title)
    if not normalized:
        return f"Slide {index}"
    return _derive_slide_title(normalized, index)


def _keyword_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    lowered = text.lower()
    for token in re.findall(r"[0-9A-Za-z]+|[\u4e00-\u9fff]{2,}", lowered):
        normalized = token.strip()
        if len(normalized) >= 2:
            tokens.add(normalized)
    for token in re.findall(r"[a-z]{2,}", lowered):
        tokens.add(token)
    return tokens


def assess_slides_payload_for_video(
    payload: dict[str, Any],
    *,
    input_video: Path,
    explicit_path: bool = False,
) -> dict[str, Any]:
    slides = payload.get("slides", [])
    if not isinstance(slides, list) or not slides:
        payload["usable"] = False
        payload["match_score"] = 0
        payload["matched_tokens"] = []
        payload["rejected_reason"] = "no_slides"
        return payload

    if explicit_path:
        payload["usable"] = True
        payload["match_score"] = len(slides)
        payload["matched_tokens"] = ["explicit_path"]
        payload["rejected_reason"] = ""
        return payload

    payload["matched_tokens"] = []
    payload["match_score"] = 0
    payload["usable"] = False
    payload["rejected_reason"] = "explicit_slides_required"
    return payload


def determine_visual_source_mode(
    *,
    requested_mode: str,
    slides_payload: dict[str, Any] | None,
) -> str:
    usable = bool((slides_payload or {}).get("usable", False))
    if requested_mode == "video-first":
        return "video-first"
    if requested_mode == "slides-first":
        return "slides-first" if usable else "video-first"
    return "slides-first" if usable else "video-first"


def parse_slides_pdf(pdf_path: Path, preview_dir: Path) -> dict[str, Any]:
    rendered_dir = preview_dir / "rendered"
    if rendered_dir.exists():
        shutil.rmtree(rendered_dir)
    rendered_dir.mkdir(parents=True, exist_ok=True)

    slides: list[dict[str, Any]] = []
    with fitz.open(pdf_path) as document:
        for index, page in enumerate(document, start=1):
            slide_text = _normalize_slide_text(page.get_text("text"))
            title = _derive_slide_title(slide_text, index)
            rendered_path = rendered_dir / f"slide_{index:03d}.png"
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
            pixmap.save(rendered_path)
            slides.append(
                {
                    "slide_id": f"slide_{index:03d}",
                    "slide_index": index,
                    "title": title,
                    "text": slide_text,
                    "relative_path": f"{preview_dir.name}/rendered/slide_{index:03d}.png",
                    "image_frequency": 1,
                    "image_area": int(pixmap.width * pixmap.height),
                    "is_low_value": (
                        len(slide_text) < 8
                        or title.lower() in {"contents", "目录", "slide"}
                        or "目录" in title
                    ),
                }
            )

    return {
        "slide_count": len(slides),
        "preview_dir": preview_dir.name,
        "slides": slides,
    }


def prepare_slides_payload(slides_path: Path, preview_dir: Path) -> dict[str, Any]:
    payload = parse_slides_pdf(slides_path, preview_dir)
    work_dir = preview_dir.parent.parent.parent
    for slide in payload.get("slides", []):
        if not isinstance(slide, dict):
            continue
        slide_index = int(slide.get("slide_index", 0) or 0)
        rendered_path = preview_dir / "rendered" / f"slide_{slide_index:03d}.png"
        if rendered_path.exists():
            slide["relative_path"] = rendered_path.relative_to(work_dir).as_posix()
    payload["payload_version"] = SLIDES_PAYLOAD_VERSION
    payload["source_file"] = slides_path.name
    payload["source_path"] = str(slides_path)
    payload["source_kind"] = "pdf"
    return payload


def sanitize_slides_payload(payload: dict[str, Any]) -> dict[str, Any]:
    slides = payload.get("slides", [])
    if not isinstance(slides, list):
        return payload

    sanitized_slides: list[dict[str, Any]] = []
    for slide in slides:
        if not isinstance(slide, dict):
            continue
        sanitized_slide = dict(slide)
        raw_text = str(sanitized_slide.get("text", "")).strip()
        raw_title = str(sanitized_slide.get("title", "")).strip()
        cleaned_text = _sanitize_slide_text(raw_text)
        cleaned_title = _compact_existing_title(_sanitize_slide_text(raw_title), int(sanitized_slide.get("slide_index", 0) or 0))
        if not cleaned_title or _text_has_slide_noise(cleaned_title) or cleaned_title.lower().startswith("slide "):
            cleaned_title = _derive_slide_title(cleaned_text, int(sanitized_slide.get("slide_index", 0) or 0))
        sanitized_slide["text"] = cleaned_text
        sanitized_slide["title"] = cleaned_title
        sanitized_slide["is_low_value"] = bool(
            len(cleaned_text) < 8
            or cleaned_title.lower() in {"contents", "目录", "slide"}
            or "目录" in cleaned_title
        )
        sanitized_slides.append(sanitized_slide)

    updated = dict(payload)
    updated["slides"] = sanitized_slides
    return updated


def slides_payload_has_noise(payload: dict[str, Any]) -> bool:
    slides = payload.get("slides", [])
    if not isinstance(slides, list):
        return False
    return any(
        _text_has_slide_noise(str(slide.get("title", "")).strip()) or _text_has_slide_noise(str(slide.get("text", "")).strip())
        for slide in slides
        if isinstance(slide, dict)
    )

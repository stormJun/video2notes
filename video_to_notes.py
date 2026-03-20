#!/usr/bin/env python3

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from video_to_notes_note import (
    build_note_blocks,
    build_note_generation_prompt,
    build_note_outline,
    render_note_markdown,
    render_ppt_alignment_debug_markdown,
    sanitize_note_body_timestamps,
)
from video_to_notes_schema import format_seconds, iso_timestamp, plan_output_paths, slugify_stem
from video_to_notes_slides import slides_payload_has_noise
from video_to_notes_visual import (
    annotate_scene_change_scores,
    ocr_quality_score,
    preprocess_image_for_ocr,
    scan_visual_candidate_timestamps,
)


def _keyword_tokens(*texts: str) -> set[str]:
    tokens: set[str] = set()
    for text in texts:
        for match in re.findall(r"[0-9A-Za-z]+|[\u4e00-\u9fff]{2,}", text.lower()):
            normalized = match.strip()
            if len(normalized) >= 2:
                tokens.add(normalized)
    return tokens


def _token_weight(token: str) -> float:
    token = token.strip().lower()
    if not token:
        return 0.0
    weights = {
        "transformer": 0.0,
        "bert": 0.5,
        "rnn": 0.25,
        "dnn": 0.25,
        "lstm": 0.5,
        "gru": 0.5,
        "cls": 0.75,
        "sep": 0.75,
        "embedding": 0.75,
        "text": 0.25,
        "representations": 0.5,
    }
    return weights.get(token, 1.0)


def _overlap_metrics(left_tokens: set[str], right_tokens: set[str]) -> tuple[int, float]:
    overlap_tokens = left_tokens & right_tokens
    weighted_score = sum(_token_weight(token) for token in overlap_tokens)
    return len(overlap_tokens), float(weighted_score)


def _segment_keywords(segment: dict[str, Any], review: dict[str, Any], cleaned_text: str) -> set[str]:
    return _keyword_tokens(
        str(review.get("summary", "")).strip(),
        " ".join(str(item).strip() for item in review.get("issues", []) if str(item).strip()),
        cleaned_text,
        str(segment.get("text", "")).strip(),
        " ".join(str(item).strip() for item in segment.get("ocr_hints", []) if str(item).strip()),
    )


def _score_frame_candidates(
    *,
    frames: list[dict[str, Any]],
    segment: dict[str, Any],
    review: dict[str, Any],
    cleaned_text: str,
    start: float,
    end: float,
) -> list[dict[str, Any]]:
    segment_keywords = _segment_keywords(segment, review, cleaned_text)
    positive_overlap_exists = False
    center = (start + end) / 2
    candidates: list[dict[str, Any]] = []

    for frame in frames:
        timestamp = float(frame.get("timestamp", 0.0) or 0.0)
        frame_text = str(frame.get("ocr_text", "")).strip()
        frame_keywords = _keyword_tokens(frame_text)
        overlap, weighted_overlap = _overlap_metrics(segment_keywords, frame_keywords)
        time_bonus = 0
        if start - 5 <= timestamp <= end + 5:
            time_bonus = 2
        elif start - 30 <= timestamp <= end + 30:
            time_bonus = 1

        if weighted_overlap > 0:
            positive_overlap_exists = True
        if weighted_overlap == 0 and time_bonus == 0:
            continue

        candidates.append(
            {
                "source": "frame",
                "score": float(weighted_overlap * 10 + time_bonus),
                "text_score": float(weighted_overlap * 10),
                "time_score": float(time_bonus),
                "quality_score": 0.0,
                "overlap": overlap,
                "time_distance_seconds": abs(timestamp - center),
                "timestamp": timestamp,
                "relative_path": str(frame.get("relative_path", "")).strip(),
                "ocr_text": frame_text,
                "is_low_value": False,
                "selection_reason": [
                    "text_overlap" if weighted_overlap > 0 else "time_proximity",
                    "raw_frame_fallback",
                ],
            }
        )

    if positive_overlap_exists:
        candidates = [item for item in candidates if float(item["text_score"]) > 0]

    candidates.sort(key=lambda item: (-float(item["score"]), float(item["time_distance_seconds"]), item["relative_path"]))
    return candidates


def _score_ppt_slide_candidates(
    *,
    slides: list[dict[str, Any]],
    segment: dict[str, Any],
    review: dict[str, Any],
    cleaned_text: str,
    start: float,
) -> list[dict[str, Any]]:
    segment_keywords = _segment_keywords(segment, review, cleaned_text)
    segment_text_bundle = " ".join(
        [
            str(review.get("summary", "")).strip(),
            cleaned_text,
            str(segment.get("text", "")).strip(),
        ]
    )
    candidates: list[dict[str, Any]] = []
    for slide in slides:
        if bool(slide.get("is_low_value", False)):
            continue
        relative_path = str(slide.get("relative_path", "")).strip()
        if not relative_path:
            continue
        slide_text = str(slide.get("text", "")).strip()
        slide_title = str(slide.get("title", "")).strip()
        title_tokens = _keyword_tokens(slide_title)
        body_tokens = _keyword_tokens(slide_text)
        title_overlap, title_weighted_overlap = _overlap_metrics(segment_keywords, title_tokens)
        body_overlap, body_weighted_overlap = _overlap_metrics(segment_keywords, body_tokens)
        slide_index = int(slide.get("slide_index", 0) or 0)
        title_exact_bonus = 0.0
        if slide_title and slide_title in segment_text_bundle:
            title_exact_bonus = 18.0
        if title_weighted_overlap <= 0 and body_weighted_overlap <= 0 and title_exact_bonus <= 0:
            continue
        title_score = title_weighted_overlap * 20
        body_score = body_weighted_overlap * 6
        corroboration_bonus = 4.0 if title_weighted_overlap > 0 and body_weighted_overlap > 0 else 0.0
        score = title_score + body_score + corroboration_bonus + title_exact_bonus
        candidates.append(
            {
                "source": "ppt_slide",
                "slide_id": str(slide.get("slide_id", "")).strip(),
                "slide_index": slide_index,
                "score": float(score),
                "base_score": float(score),
                "text_score": float(title_score + body_score + corroboration_bonus + title_exact_bonus),
                "time_score": 0.0,
                "quality_score": 6.0,
                "overlap": title_overlap + body_overlap,
                "time_distance_seconds": 0.0,
                "timestamp": float(start),
                "relative_path": relative_path,
                "ocr_text": slide_text,
                "is_low_value": False,
                "slide_title": slide_title,
                "sequence_score": 0.0,
                "selection_reason": [
                    "ppt_slide_match",
                    "title_exact_match" if title_exact_bonus > 0 else "no_title_exact_match",
                    "title_overlap" if title_weighted_overlap > 0 else "body_overlap",
                    "body_overlap" if body_weighted_overlap > 0 else "title_only_match",
                    "sequence_unscored_base_candidate",
                ],
            }
        )
    candidates.sort(key=lambda item: (-float(item["score"]), int(item.get("slide_index", 0) or 0)))
    return candidates


def _ppt_opening_score(slide_index: int, base_score: float) -> tuple[float, str]:
    opening_score = -float(max(slide_index - 1, 0)) * 0.6
    reason = "opening_sequence_preference"
    if base_score >= 45.0 and slide_index <= 3:
        opening_score += 3.0
        reason = "opening_band_earliest_preference"
    return opening_score, reason


def _ppt_transition_score(previous_slide_index: int, slide_index: int) -> tuple[float, str]:
    delta = slide_index - previous_slide_index
    if delta < -1:
        return -34.0 - abs(delta) * 3.0, "sequence_backtrack_penalty"
    if delta == -1:
        return -12.0, "sequence_step_back_penalty"
    if delta == 0:
        return 4.0, "sequence_hold_bonus"
    if delta == 1:
        return 10.0, "sequence_advance_bonus"
    if delta <= 3:
        return 7.0 - float(delta - 1), "sequence_nearby_bonus"
    if delta <= 8:
        return 1.5 - float(delta - 4) * 0.5, "sequence_far_forward"
    return -4.0 - float(delta - 8) * 1.5, "sequence_large_jump_penalty"


def _select_ppt_visual_sequence(
    candidate_rows: list[list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]]]:
    if not candidate_rows:
        return [], []

    score_rows: list[list[float]] = []
    backpointers: list[list[int | None]] = []
    reason_rows: list[list[str]] = []
    step_score_rows: list[list[float]] = []
    previous_slide_rows: list[list[int | None]] = []

    first_row_scores: list[float] = []
    first_row_backpointers: list[int | None] = []
    first_row_reasons: list[str] = []
    first_row_step_scores: list[float] = []
    first_row_previous_slides: list[int | None] = []
    for candidate in candidate_rows[0]:
        slide_index = int(candidate.get("slide_index", 0) or 0)
        opening_score, opening_reason = _ppt_opening_score(slide_index, float(candidate.get("score", 0.0)))
        first_row_scores.append(float(candidate.get("score", 0.0)) + opening_score)
        first_row_backpointers.append(None)
        first_row_reasons.append(opening_reason)
        first_row_step_scores.append(float(opening_score))
        first_row_previous_slides.append(None)
    score_rows.append(first_row_scores)
    backpointers.append(first_row_backpointers)
    reason_rows.append(first_row_reasons)
    step_score_rows.append(first_row_step_scores)
    previous_slide_rows.append(first_row_previous_slides)

    for row_index in range(1, len(candidate_rows)):
        row_scores: list[float] = []
        row_backpointers: list[int | None] = []
        row_reasons: list[str] = []
        row_step_scores: list[float] = []
        row_previous_slides: list[int | None] = []
        for candidate in candidate_rows[row_index]:
            slide_index = int(candidate.get("slide_index", 0) or 0)
            best_score = float("-inf")
            best_prev_index: int | None = None
            best_reason = "sequence_unreachable"
            best_step_score = 0.0
            best_previous_slide_index: int | None = None
            for previous_candidate_index, previous_candidate in enumerate(candidate_rows[row_index - 1]):
                previous_slide_index = int(previous_candidate.get("slide_index", 0) or 0)
                transition_score, transition_reason = _ppt_transition_score(previous_slide_index, slide_index)
                total_score = score_rows[row_index - 1][previous_candidate_index] + float(candidate.get("score", 0.0)) + transition_score
                if total_score > best_score:
                    best_score = total_score
                    best_prev_index = previous_candidate_index
                    best_reason = transition_reason
                    best_step_score = float(transition_score)
                    best_previous_slide_index = previous_slide_index
            row_scores.append(best_score)
            row_backpointers.append(best_prev_index)
            row_reasons.append(best_reason)
            row_step_scores.append(best_step_score)
            row_previous_slides.append(best_previous_slide_index)
        score_rows.append(row_scores)
        backpointers.append(row_backpointers)
        reason_rows.append(row_reasons)
        step_score_rows.append(row_step_scores)
        previous_slide_rows.append(row_previous_slides)

    final_row = score_rows[-1]
    best_final_index = max(
        range(len(final_row)),
        key=lambda idx: (final_row[idx], -int(candidate_rows[-1][idx].get("slide_index", 0) or 0)),
    )
    suffix_extra_rows: list[list[float]] = [[0.0 for _ in row] for row in candidate_rows]
    for row_index in range(len(candidate_rows) - 2, -1, -1):
        next_row = candidate_rows[row_index + 1]
        for candidate_index, candidate in enumerate(candidate_rows[row_index]):
            slide_index = int(candidate.get("slide_index", 0) or 0)
            best_extra = float("-inf")
            for next_candidate_index, next_candidate in enumerate(next_row):
                next_slide_index = int(next_candidate.get("slide_index", 0) or 0)
                transition_score, _ = _ppt_transition_score(slide_index, next_slide_index)
                extra_score = (
                    transition_score
                    + float(next_candidate.get("base_score", next_candidate.get("score", 0.0)))
                    + suffix_extra_rows[row_index + 1][next_candidate_index]
                )
                if extra_score > best_extra:
                    best_extra = extra_score
            suffix_extra_rows[row_index][candidate_index] = 0.0 if best_extra == float("-inf") else best_extra

    annotated_rows: list[list[dict[str, Any]]] = []
    for row_index, row in enumerate(candidate_rows):
        annotated_row: list[dict[str, Any]] = []
        for candidate_index, candidate in enumerate(row):
            annotated = dict(candidate)
            annotated["sequence_score"] = float(step_score_rows[row_index][candidate_index])
            annotated["cumulative_score"] = float(score_rows[row_index][candidate_index])
            annotated["global_path_score"] = float(
                score_rows[row_index][candidate_index] + suffix_extra_rows[row_index][candidate_index]
            )
            annotated["sequence_previous_slide_index"] = previous_slide_rows[row_index][candidate_index]
            annotated["sequence_reason"] = reason_rows[row_index][candidate_index]
            annotated_row.append(annotated)
        annotated_rows.append(annotated_row)

    selected: list[dict[str, Any]] = []
    current_index: int | None = best_final_index
    for row_index in range(len(candidate_rows) - 1, -1, -1):
        if current_index is None:
            break
        chosen = dict(annotated_rows[row_index][current_index])
        chosen["selection_reason"] = list(chosen.get("selection_reason", [])) + [
            reason_rows[row_index][current_index],
            "global_sequence_optimization",
        ]
        chosen["sequence_selected"] = True
        selected.append(chosen)
        current_index = backpointers[row_index][current_index]

    selected.reverse()
    selected_by_row = {
        row_index: (selected[row_index] if row_index < len(selected) else None)
        for row_index in range(len(annotated_rows))
    }
    for row_index, annotated_row in enumerate(annotated_rows):
        selected_item = selected_by_row.get(row_index)
        selected_slide_id = str((selected_item or {}).get("slide_id", "")).strip()
        selected_global_path_score = float((selected_item or {}).get("global_path_score", float("-inf")))
        for candidate in annotated_row:
            is_selected = str(candidate.get("slide_id", "")).strip() == selected_slide_id and selected_slide_id != ""
            candidate["sequence_selected"] = is_selected
            if is_selected:
                candidate["selection_reason"] = list(candidate.get("selection_reason", [])) + [
                    str(candidate.get("sequence_reason", "")).strip(),
                    "global_sequence_optimization",
                ]
                candidate["sequence_rejection_reason"] = ""
                candidate["sequence_score_gap"] = 0.0
            else:
                score_gap = selected_global_path_score - float(candidate.get("global_path_score", float("-inf")))
                candidate["sequence_rejection_reason"] = (
                    f"lost_to_slide_{(selected_item or {}).get('slide_index', 0):03d}_on_global_path_score"
                    if selected_slide_id
                    else "not_selected_by_global_sequence"
                )
                candidate["sequence_score_gap"] = float(score_gap)
    selected = []
    for row_index, annotated_row in enumerate(annotated_rows):
        selected_slide_id = str((selected_by_row.get(row_index) or {}).get("slide_id", "")).strip()
        if not selected_slide_id:
            continue
        for candidate in annotated_row:
            if str(candidate.get("slide_id", "")).strip() == selected_slide_id:
                selected.append(dict(candidate))
                break
    return selected, annotated_rows


def _score_visual_unit_candidates(
    *,
    visual_units: list[dict[str, Any]],
    segment: dict[str, Any],
    review: dict[str, Any],
    cleaned_text: str,
    start: float,
    end: float,
) -> tuple[list[dict[str, Any]], str]:
    segment_keywords = _segment_keywords(segment, review, cleaned_text)
    center = (start + end) / 2

    def collect(window_seconds: float | None) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        positive_overlap_exists = False
        non_low_value_exists = False
        for unit in visual_units:
            timestamp = float(unit.get("representative_timestamp", unit.get("start", 0.0)) or 0.0)
            ocr_text = str(unit.get("ocr_text", "")).strip()
            overlap, weighted_overlap = _overlap_metrics(segment_keywords, _keyword_tokens(ocr_text))
            is_low_value = bool(unit.get("is_low_value", False))
            if window_seconds is not None and abs(timestamp - center) > window_seconds:
                continue
            if window_seconds is None and weighted_overlap == 0:
                continue
            if window_seconds is None and weighted_overlap < 1.5:
                continue
            if weighted_overlap > 0:
                positive_overlap_exists = True
            if not is_low_value:
                non_low_value_exists = True

            time_penalty = abs(timestamp - center) / 10
            quality_penalty = 8 if is_low_value else 0
            score = weighted_overlap * 10 - time_penalty - quality_penalty
            if score <= 0 and weighted_overlap == 0:
                continue
            candidates.append(
                {
                    "source": "visual_unit",
                    "unit_id": str(unit.get("unit_id", "")).strip(),
                    "score": float(score),
                    "text_score": float(weighted_overlap * 10),
                    "time_score": float(-time_penalty),
                    "quality_score": float(-quality_penalty),
                    "overlap": overlap,
                    "time_distance_seconds": abs(timestamp - center),
                    "timestamp": timestamp,
                    "relative_path": str(unit.get("representative_frame", "")).strip(),
                    "ocr_text": ocr_text,
                    "is_low_value": is_low_value,
                    "selection_reason": [
                        "nearby_window" if window_seconds is not None else "cross_segment_borrow",
                        "text_overlap" if weighted_overlap > 0 else "time_proximity",
                        "visual_unit_match",
                    ],
                }
            )

        if positive_overlap_exists:
            candidates = [item for item in candidates if float(item["text_score"]) > 0]
        if non_low_value_exists:
            candidates = [item for item in candidates if not bool(item["is_low_value"])]
        candidates.sort(key=lambda item: (-float(item["score"]), float(item["time_distance_seconds"]), item["relative_path"]))
        return candidates

    nearby = collect(45.0)
    if nearby:
        return nearby, "visual_unit_nearby"
    return collect(None), "visual_unit_global"


def _borrow_nearby_high_quality_visuals(
    *,
    visual_units: list[dict[str, Any]],
    segment: dict[str, Any],
    review: dict[str, Any],
    cleaned_text: str,
    start: float,
    end: float,
    max_window_seconds: float = 120.0,
) -> list[dict[str, Any]]:
    center = (start + end) / 2
    segment_keywords = _segment_keywords(segment, review, cleaned_text)
    candidates: list[dict[str, Any]] = []
    for unit in visual_units:
        if bool(unit.get("is_low_value", False)):
            continue
        timestamp = float(unit.get("representative_timestamp", unit.get("start", 0.0)) or 0.0)
        distance = abs(timestamp - center)
        if distance > max_window_seconds:
            continue
        overlap, weighted_overlap = _overlap_metrics(segment_keywords, _keyword_tokens(str(unit.get("ocr_text", "")).strip()))
        if weighted_overlap < 0.75:
            continue
        candidates.append(
            {
                "source": "visual_unit",
                "unit_id": str(unit.get("unit_id", "")).strip(),
                "score": float(weighted_overlap * 12 + max_window_seconds - distance + 5),
                "text_score": float(weighted_overlap * 12),
                "time_score": float(max_window_seconds - distance),
                "quality_score": 5.0,
                "overlap": overlap,
                "time_distance_seconds": distance,
                "timestamp": timestamp,
                "relative_path": str(unit.get("representative_frame", "")).strip(),
                "ocr_text": str(unit.get("ocr_text", "")).strip(),
                "is_low_value": False,
                "selection_reason": [
                    "nearby_high_quality_borrow",
                    "text_overlap",
                    "time_proximity",
                    "visual_unit_match",
                ],
            }
        )
    candidates.sort(key=lambda item: (-float(item["score"]), float(item["time_distance_seconds"]), item["relative_path"]))
    return candidates[:2]


def build_visual_alignment(
    *,
    review_segments: list[dict[str, Any]] | None = None,
    segment_reviews: list[dict[str, Any]] | None = None,
    cleaned_segments: list[str] | None = None,
    visual_units: list[dict[str, Any]] | None = None,
    slides: list[dict[str, Any]] | None = None,
    frames: list[dict[str, Any]] | None = None,
    visual_source_mode: str = "auto",
) -> dict[str, Any]:
    review_segments = review_segments or []
    segment_reviews = segment_reviews or []
    cleaned_segments = cleaned_segments or []
    visual_units = visual_units or []
    slides = slides or []
    frames = frames or []
    effective_visual_source_mode = (
        visual_source_mode
        if visual_source_mode != "auto"
        else ("slides-first" if slides else "video-first")
    )

    segment_map = {str(item.get("segment_id", "")): item for item in review_segments}
    cleaned_segment_map = {
        str(segment.get("segment_id", "")).strip(): cleaned_segments[index].strip()
        for index, segment in enumerate(review_segments)
        if index < len(cleaned_segments) and str(segment.get("segment_id", "")).strip() and cleaned_segments[index].strip()
    }
    cleaned_segment_reviews = [
        item
        for item in segment_reviews
        if isinstance(item, dict) and str(item.get("status", "")).strip() == "done" and str(item.get("summary", "")).strip()
    ]

    optimized_ppt_sequence: list[dict[str, Any]] = []
    optimized_ppt_candidates_by_segment: dict[str, list[dict[str, Any]]] = {}
    ppt_sequence_trace_by_segment: dict[str, dict[str, Any]] = {}
    if slides and effective_visual_source_mode == "slides-first":
        ppt_candidate_rows: list[list[dict[str, Any]]] = []
        ppt_segment_ids: list[str] = []
        for review in cleaned_segment_reviews:
            segment_id = str(review.get("segment_id", "")).strip()
            segment = segment_map.get(segment_id, {})
            start = float(review.get("start", segment.get("start", 0.0)) or 0.0)
            cleaned_text = cleaned_segment_map.get(segment_id, "").strip()
            raw_text = str(segment.get("text", "")).strip()
            candidates = _score_ppt_slide_candidates(
                slides=slides,
                segment=segment,
                review=review,
                cleaned_text=cleaned_text or raw_text,
                start=start,
            )
            if candidates:
                ppt_candidate_rows.append(candidates[:8])
                ppt_segment_ids.append(segment_id)
        if ppt_candidate_rows:
            optimized_ppt_sequence, annotated_ppt_rows = _select_ppt_visual_sequence(ppt_candidate_rows)
            optimized_ppt_candidates_by_segment = {
                segment_id: candidates
                for segment_id, candidates in zip(ppt_segment_ids, annotated_ppt_rows, strict=False)
            }
            optimized_ppt_selected_by_segment = {
                segment_id: selected
                for segment_id, selected in zip(ppt_segment_ids, optimized_ppt_sequence, strict=False)
            }
            ppt_sequence_trace_by_segment = {
                segment_id: {
                    "selected_slide_id": str(selected.get("slide_id", "")).strip(),
                    "selected_slide_index": int(selected.get("slide_index", 0) or 0),
                    "base_score": float(selected.get("base_score", selected.get("score", 0.0)) or 0.0),
                    "sequence_step_score": float(selected.get("sequence_score", 0.0) or 0.0),
                    "cumulative_score": float(selected.get("cumulative_score", selected.get("score", 0.0)) or 0.0),
                    "global_path_score": float(selected.get("global_path_score", selected.get("score", 0.0)) or 0.0),
                    "previous_slide_index": selected.get("sequence_previous_slide_index"),
                    "sequence_reason": str(selected.get("sequence_reason", "")).strip(),
                }
                for segment_id, selected in zip(ppt_segment_ids, optimized_ppt_sequence, strict=False)
            }
        else:
            optimized_ppt_selected_by_segment = {}
    else:
        optimized_ppt_selected_by_segment = {}

    aligned_segments: list[dict[str, Any]] = []
    useful_visual_count = 0
    for review in cleaned_segment_reviews:
        segment_id = str(review.get("segment_id", "")).strip()
        segment = segment_map.get(segment_id, {})
        start = float(review.get("start", segment.get("start", 0.0)) or 0.0)
        end = float(review.get("end", segment.get("end", start)) or start)
        label = str(segment.get("label", "")).strip() or f"{format_seconds(start)}-{format_seconds(end)}"
        cleaned_text = cleaned_segment_map.get(segment_id, "").strip()
        raw_text = str(segment.get("text", "")).strip()
        issues = [str(item).strip() for item in review.get("issues", []) if str(item).strip()]
        reject_reason = ""
        borrow_reason = ""

        ppt_candidates = optimized_ppt_candidates_by_segment.get(segment_id, [])
        if ppt_candidates:
            selected_ppt = optimized_ppt_selected_by_segment.get(segment_id)
            if selected_ppt:
                visual_candidates = [selected_ppt] + [
                    candidate
                    for candidate in ppt_candidates
                    if str(candidate.get("slide_id", "")).strip() != str(selected_ppt.get("slide_id", "")).strip()
                ]
            else:
                visual_candidates = ppt_candidates
            selection_mode = "ppt_slide_match"
        else:
            visual_candidates, selection_mode = _score_visual_unit_candidates(
                visual_units=visual_units,
                segment=segment,
                review=review,
                cleaned_text=cleaned_text or raw_text,
                start=start,
                end=end,
            )
        selected_visuals = visual_candidates[:1]
        fallback_candidates: list[dict[str, Any]] = []
        if selection_mode != "ppt_slide_match" and selected_visuals and all(bool(item.get("is_low_value", False)) for item in selected_visuals):
            selected_visuals = []
            selection_mode = "suppressed_low_value"
            reject_reason = "nearby matched visual units were all low-value"
            borrowed_candidates = _borrow_nearby_high_quality_visuals(
                visual_units=visual_units,
                segment=segment,
                review=review,
                cleaned_text=cleaned_text or raw_text,
                start=start,
                end=end,
            )
            if borrowed_candidates:
                selected_visuals = borrowed_candidates
                selection_mode = "borrowed_nearby_visual"
                borrow_reason = "borrowed nearby non-low-value visual units that also met minimum semantic overlap"
                reject_reason = ""
            else:
                reject_reason = "nearby matched visual units were all low-value and no nearby high-quality visual met semantic overlap"
        if not selected_visuals and not visual_candidates:
            fallback_candidates = _score_frame_candidates(
                frames=frames,
                segment=segment,
                review=review,
                cleaned_text=cleaned_text or raw_text,
                start=start,
                end=end,
            )
            selected_visuals = fallback_candidates[:1]
            selection_mode = "frame_fallback" if selected_visuals else "none"
            if selected_visuals:
                borrow_reason = "no visual-unit match, fell back to raw frames"
            else:
                reject_reason = "no visual candidate passed alignment thresholds"

        if selected_visuals:
            useful_visual_count += 1

        aligned_segments.append(
            {
                "segment_id": segment_id,
                "label": label,
                "start": start,
                "end": end,
                "summary": str(review.get("summary", "")).strip(),
                "issues": issues,
                "cleaned_text": cleaned_text,
                "raw_text": raw_text,
                "selection_mode": selection_mode,
                "has_useful_visual": bool(selected_visuals),
                "borrow_reason": borrow_reason,
                "reject_reason": reject_reason,
                "ppt_sequence_trace": ppt_sequence_trace_by_segment.get(segment_id, {}),
                "selected_visuals": selected_visuals,
                "candidate_visuals": (visual_candidates or fallback_candidates)[:5],
            }
        )

    return {
        "segment_count": len(aligned_segments),
        "segments_with_visuals": useful_visual_count,
        "visual_source_mode": effective_visual_source_mode,
        "ppt_slide_count": len(slides),
        "visual_unit_count": len(visual_units),
        "frame_count": len(frames),
        "segments": aligned_segments,
    }


def build_note_markdown(
    *,
    title: str,
    source_video: str,
    duration_seconds: float,
    transcript_excerpt: list[str],
    cleaned_segments: list[str] | None = None,
    review_segments: list[dict[str, Any]] | None = None,
    segment_reviews: list[dict[str, Any]] | None = None,
    corrections: list[dict[str, Any]] | None = None,
    visual_units: list[dict[str, Any]] | None = None,
    visual_alignment: dict[str, Any] | None = None,
    note_outline: dict[str, Any] | None = None,
    note_blocks: dict[str, Any] | None = None,
    frames: list[dict[str, Any]],
) -> str:
    cleaned_segments = cleaned_segments or []
    review_segments = review_segments or []
    segment_reviews = segment_reviews or []
    corrections = corrections or []
    visual_units = visual_units or []
    note_outline = note_outline or {}
    note_blocks = note_blocks or {}
    visual_alignment = visual_alignment or build_visual_alignment(
        review_segments=review_segments,
        segment_reviews=segment_reviews,
        cleaned_segments=cleaned_segments,
        visual_units=visual_units,
        frames=frames,
    )
    if not note_outline:
        note_outline = build_note_outline(title=title, visual_alignment=visual_alignment)
    if not note_blocks:
        note_blocks = build_note_blocks(note_outline=note_outline, visual_alignment=visual_alignment)
    return sanitize_note_body_timestamps(
        render_note_markdown(
        title=title,
        source_video=source_video,
        duration_seconds=duration_seconds,
        transcript_excerpt=transcript_excerpt,
        note_outline=note_outline,
        note_blocks=note_blocks,
        corrections=corrections,
        frames=frames,
        )
    )


def explain_note_generation_failure(note_path: Path) -> list[str]:
    reasons: list[str] = []
    if not note_path.exists():
        return ["note.md 不存在"]
    text = note_path.read_text(encoding="utf-8").strip()
    if not text:
        return ["note.md 为空"]
    if "## 知识小结" not in text:
        reasons.append("note.md 缺少知识小结")
    if "核心定义卡片" not in text:
        reasons.append("note.md 缺少核心定义卡片")
    if "知识框架" not in text:
        reasons.append("note.md 缺少知识框架")
    if not re.search(r"\[\d{2}:\d{2}(?::\d{2})?\]", text):
        reasons.append("note.md 缺少精确时间戳引用")
    return reasons


def note_generation_completed(note_path: Path) -> bool:
    return not explain_note_generation_failure(note_path)


def relative_output_path(output_paths: dict[str, Path | str], key: str) -> str:
    work_dir = Path(output_paths["work_dir"])
    target = Path(output_paths[key])
    try:
        return target.relative_to(work_dir).as_posix()
    except ValueError:
        return str(target)


def build_note_generation_report(
    *,
    output_paths: dict[str, Path | str],
    generator: str,
    attempts: list[dict[str, Any]],
    status: str,
    final_failure_reasons: list[str],
) -> dict[str, Any]:
    note_path = Path(output_paths["note_path"])
    prompt_path = Path(output_paths["note_generation_prompt_md"])
    outline_path = Path(output_paths["note_outline_json"])
    blocks_path = Path(output_paths["note_blocks_json"])
    quality_gate_failures = [] if status == "passed" else list(final_failure_reasons or explain_note_generation_failure(note_path))
    return {
        "note_file": relative_output_path(output_paths, "note_path"),
        "prompt_file": relative_output_path(output_paths, "note_generation_prompt_md"),
        "note_outline_file": relative_output_path(output_paths, "note_outline_json"),
        "note_blocks_file": relative_output_path(output_paths, "note_blocks_json"),
        "status": status,
        "generator": generator,
        "attempt_count": len(attempts),
        "attempts": attempts,
        "quality_gate_passed": status == "passed" and not quality_gate_failures,
        "quality_gate_failures": quality_gate_failures,
        "final_failure_reasons": final_failure_reasons,
        "last_updated": iso_timestamp(),
        "work_dir": str(output_paths["work_dir"]),
    }


def write_note_generation_report(
    output_paths: dict[str, Path | str],
    *,
    generator: str,
    attempts: list[dict[str, Any]],
    status: str,
    final_failure_reasons: list[str],
) -> None:
    Path(output_paths["note_generation_report_json"]).write_text(
        json.dumps(
            build_note_generation_report(
                output_paths=output_paths,
                generator=generator,
                attempts=attempts,
                status=status,
                final_failure_reasons=final_failure_reasons,
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def build_slides_cleanup_report(
    *,
    output_paths: dict[str, Path | str],
    generator: str,
    attempts: list[dict[str, Any]],
    status: str,
    final_failure_reasons: list[str],
    noise_detected_before: bool,
    noise_detected_after: bool,
) -> dict[str, Any]:
    return {
        "slides_raw_file": relative_output_path(output_paths, "slides_index_raw_json"),
        "slides_file": relative_output_path(output_paths, "slides_index_json"),
        "prompt_file": relative_output_path(output_paths, "slides_cleanup_prompt_md"),
        "status": status,
        "generator": generator,
        "attempt_count": len(attempts),
        "attempts": attempts,
        "noise_detected_before": noise_detected_before,
        "noise_detected_after": noise_detected_after,
        "final_failure_reasons": final_failure_reasons,
        "last_updated": iso_timestamp(),
        "work_dir": str(output_paths["work_dir"]),
    }


def write_slides_cleanup_report(
    output_paths: dict[str, Path | str],
    *,
    generator: str,
    attempts: list[dict[str, Any]],
    status: str,
    final_failure_reasons: list[str],
    noise_detected_before: bool,
    noise_detected_after: bool,
) -> None:
    Path(output_paths["slides_cleanup_report_json"]).write_text(
        json.dumps(
            build_slides_cleanup_report(
                output_paths=output_paths,
                generator=generator,
                attempts=attempts,
                status=status,
                final_failure_reasons=final_failure_reasons,
                noise_detected_before=noise_detected_before,
                noise_detected_after=noise_detected_after,
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def build_slides_cleanup_prompt(output_paths: dict[str, Path | str]) -> str:
    slides_raw_file = relative_output_path(output_paths, "slides_index_raw_json")
    slides_file = relative_output_path(output_paths, "slides_index_json")
    slides_preview_dir = relative_output_path(output_paths, "slides_preview_dir")
    return "\n".join(
        [
            f"请清洗当前目录中的课件页文本索引，只允许改动 `{slides_file}`。",
            "",
            "输入：",
            f"- 原始课件索引：`{slides_raw_file}`",
            f"- 当前可编辑索引：`{slides_file}`",
            f"- 课件页图片目录：`{slides_preview_dir}/rendered/`",
            "",
            "你的任务：",
            f"1. 阅读 `{slides_raw_file}`、`{slides_file}`，必要时查看 `{slides_preview_dir}/rendered/slide_XXX.png`。",
            "2. 只修正每页的 `title`、`text`、`is_low_value`。",
            "3. 删除或改写明显乱码，例如 `ǜ֩`、`֩`、错误的下标碎片。",
            "4. 保留真实可见的中文/英文标题与正文，不要编造课件里没有的信息。",
            "5. 若某页文本主要是公式或符号且无法可靠恢复，可保留简短可读标题，不要输出乱码。",
            "",
            "硬约束：",
            "- 不要修改 `slide_id`、`slide_index`、`relative_path`、`image_area`、`image_frequency`、`slide_count`、`source_path` 等结构字段。",
            "- 不要修改任何图片文件。",
            f"- 只允许改动 `{slides_file}`。",
        ]
    )


def build_slides_cleanup_agent_instructions(output_paths: dict[str, Path | str]) -> str:
    slides_raw_file = relative_output_path(output_paths, "slides_index_raw_json")
    slides_file = relative_output_path(output_paths, "slides_index_json")
    slides_preview_dir = relative_output_path(output_paths, "slides_preview_dir")
    return "\n".join(
        [
            "# AGENTS.md",
            "",
            "## Scope",
            "",
            "当前目录只允许清洗课件页文本索引，不要运行项目脚本、不要运行测试。",
            "",
            "## Required Inputs",
            "",
            f"- `{slides_raw_file}`",
            f"- `{slides_file}`",
            f"- `{slides_preview_dir}/rendered/`",
            "",
            "## Allowed Writes",
            "",
            f"- `{slides_file}`",
            "",
            "## Hard Rules",
            "",
            f"- 只允许改动 `{slides_file}`。",
            "- 不要修改页图或其他 JSON/Markdown 文件。",
            "- 不要删除 slide 条目，不要改动 `slide_id`、`slide_index`、`relative_path`。",
        ]
    )


def _slides_cleanup_completed(raw_payload: dict[str, Any], cleaned_payload: dict[str, Any]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    raw_slides = raw_payload.get("slides", [])
    cleaned_slides = cleaned_payload.get("slides", [])
    if not isinstance(raw_slides, list) or not isinstance(cleaned_slides, list):
        return False, ["slides payload 缺少 slides 列表"]
    if len(raw_slides) != len(cleaned_slides):
        return False, ["slides 条目数量变化"]

    raw_noise = 0
    cleaned_noise = 0
    for raw_slide, cleaned_slide in zip(raw_slides, cleaned_slides):
        if not isinstance(raw_slide, dict) or not isinstance(cleaned_slide, dict):
            failures.append("slides 条目结构无效")
            continue
        for key in ("slide_id", "slide_index", "relative_path"):
            if raw_slide.get(key) != cleaned_slide.get(key):
                failures.append(f"{key} 被意外修改")
        if slides_payload_has_noise({"slides": [raw_slide]}):
            raw_noise += 1
        if slides_payload_has_noise({"slides": [cleaned_slide]}):
            cleaned_noise += 1
    if failures:
        return False, failures
    if cleaned_noise > raw_noise:
        return False, ["清洗后乱码页数不降反升"]
    return True, []


def run_codex_slides_cleanup(
    output_paths: dict[str, Path | str],
    *,
    raw_payload: dict[str, Any],
    timeout_seconds: int | None = 300,
) -> dict[str, Any]:
    work_dir = Path(output_paths["work_dir"])
    slides_path = Path(output_paths["slides_index_json"])
    prompt_path = Path(output_paths["slides_cleanup_prompt_md"])
    agents_path = work_dir / "AGENTS.md"
    prompt_text = build_slides_cleanup_prompt(output_paths)
    prompt_path.write_text(prompt_text, encoding="utf-8")

    attempts: list[dict[str, Any]] = []
    original_agents = agents_path.read_text(encoding="utf-8") if agents_path.exists() else None
    agents_path.write_text(build_slides_cleanup_agent_instructions(output_paths), encoding="utf-8")
    noise_detected_before = slides_payload_has_noise(raw_payload)
    current_payload = json.loads(slides_path.read_text(encoding="utf-8"))

    try:
        attempt_record: dict[str, Any] = {
            "attempt": 1,
            "prompt_type": "initial",
            "command_status": "started",
            "failure_reasons": [],
            "detail_excerpt": "",
            "quality_gate_passed": False,
        }
        try:
            result = run_command(
                [
                    "codex",
                    "exec",
                    "--full-auto",
                    "-c",
                    'model_reasoning_effort="low"',
                    "--skip-git-repo-check",
                    prompt_text,
                ],
                cwd=work_dir,
                timeout=timeout_seconds,
            )
        except subprocess.CalledProcessError as exc:
            stdout = exc.output.decode("utf-8", errors="ignore") if isinstance(exc.output, bytes) else (exc.output or "")
            stderr = exc.stderr.decode("utf-8", errors="ignore") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            attempt_record["command_status"] = "nonzero_exit"
            attempt_record["detail_excerpt"] = (stdout.strip() or stderr.strip())[:500]
            try:
                current_payload = json.loads(slides_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                attempt_record["failure_reasons"] = ["codex slides cleanup 返回非零退出码"]
            else:
                ok, failures = _slides_cleanup_completed(raw_payload, current_payload)
                if ok:
                    attempt_record["command_status"] = "nonzero_exit_with_valid_cleanup"
                    attempt_record["quality_gate_passed"] = True
                    attempt_record["failure_reasons"] = []
                    attempts.append(attempt_record)
                    write_slides_cleanup_report(
                        output_paths,
                        generator="codex",
                        attempts=attempts,
                        status="passed",
                        final_failure_reasons=[],
                        noise_detected_before=noise_detected_before,
                        noise_detected_after=slides_payload_has_noise(current_payload),
                    )
                    return current_payload
                attempt_record["failure_reasons"] = failures or ["codex slides cleanup 返回非零退出码"]
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout.decode("utf-8", errors="ignore") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode("utf-8", errors="ignore") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            attempt_record["command_status"] = "timeout"
            attempt_record["detail_excerpt"] = (stdout.strip() or stderr.strip())[:500]
            try:
                current_payload = json.loads(slides_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                attempt_record["failure_reasons"] = [f"codex slides cleanup 超时（>{timeout_seconds} 秒）"]
            else:
                ok, failures = _slides_cleanup_completed(raw_payload, current_payload)
                if ok:
                    attempt_record["command_status"] = "timeout_with_valid_cleanup"
                    attempt_record["quality_gate_passed"] = True
                    attempt_record["failure_reasons"] = []
                    attempts.append(attempt_record)
                    write_slides_cleanup_report(
                        output_paths,
                        generator="codex",
                        attempts=attempts,
                        status="passed",
                        final_failure_reasons=[],
                        noise_detected_before=noise_detected_before,
                        noise_detected_after=slides_payload_has_noise(current_payload),
                    )
                    return current_payload
                timeout_reason = f"codex slides cleanup 超时（>{timeout_seconds} 秒）"
                attempt_record["failure_reasons"] = failures or []
                if timeout_reason not in attempt_record["failure_reasons"]:
                    attempt_record["failure_reasons"].append(timeout_reason)
        else:
            attempt_record["command_status"] = "completed"
            attempt_record["detail_excerpt"] = (result.stdout.strip() or result.stderr.strip())[:500]
            try:
                current_payload = json.loads(slides_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                attempt_record["command_status"] = "invalid_json"
                attempt_record["failure_reasons"] = ["slides_index.json 不是有效 JSON"]
            else:
                ok, failures = _slides_cleanup_completed(raw_payload, current_payload)
                attempt_record["quality_gate_passed"] = ok
                attempt_record["failure_reasons"] = failures
                attempts.append(attempt_record)
                write_slides_cleanup_report(
                    output_paths,
                    generator="codex" if ok else "codex_fallback_rule_based",
                    attempts=attempts,
                    status="passed" if ok else "failed",
                    final_failure_reasons=failures,
                    noise_detected_before=noise_detected_before,
                    noise_detected_after=slides_payload_has_noise(current_payload),
                )
                return current_payload

        attempts.append(attempt_record)
        write_slides_cleanup_report(
            output_paths,
            generator="codex_fallback_rule_based",
            attempts=attempts,
            status="failed",
            final_failure_reasons=attempt_record["failure_reasons"],
            noise_detected_before=noise_detected_before,
            noise_detected_after=slides_payload_has_noise(current_payload),
        )
        return current_payload
    finally:
        if original_agents is None:
            agents_path.unlink(missing_ok=True)
        else:
            agents_path.write_text(original_agents, encoding="utf-8")


def build_note_codex_exec_prompt(output_paths: dict[str, Path | str]) -> str:
    note_prompt = relative_output_path(output_paths, "note_generation_prompt_md")
    outline_file = relative_output_path(output_paths, "note_outline_json")
    blocks_file = relative_output_path(output_paths, "note_blocks_json")
    cleaned_transcript = relative_output_path(output_paths, "transcript_cleaned_txt")
    raw_transcript = relative_output_path(output_paths, "transcript_txt")
    candidates_ocr = relative_output_path(output_paths, "visual_candidates_ocr_json")
    note_file = relative_output_path(output_paths, "note_path")
    corrections_file = relative_output_path(output_paths, "transcript_corrections_json")
    review_segments_file = relative_output_path(output_paths, "review_segments_json")
    metadata_file = relative_output_path(output_paths, "metadata_json")
    return "\n".join(
        [
            f"请阅读当前目录中的 `{note_prompt}` 并严格执行。",
            "你的任务只包括：",
            f"1. 阅读 `{note_prompt}`、`{outline_file}`、`{blocks_file}`、`{cleaned_transcript}`/`{raw_transcript}`、`{candidates_ocr}`。",
            f"2. 只生成或覆盖 `{note_file}`。",
            "3. 输出必须包含：课程目录、知识小结表格、核心定义卡片、知识框架、精确时间戳引用、必要时的 LaTeX 公式。",
            "限制：",
            f"- 不要修改 `{raw_transcript}`、`{cleaned_transcript}`、`{corrections_file}`、`{review_segments_file}`、`{candidates_ocr}`、`{outline_file}`、`{blocks_file}`、`{metadata_file}`。",
            f"- 只允许改动 `{note_file}`。",
            "- 如果 OCR 证据不足，不要编造公式。",
        ]
    )


def build_note_agent_instructions(output_paths: dict[str, Path | str]) -> str:
    note_prompt = relative_output_path(output_paths, "note_generation_prompt_md")
    outline_file = relative_output_path(output_paths, "note_outline_json")
    blocks_file = relative_output_path(output_paths, "note_blocks_json")
    cleaned_transcript = relative_output_path(output_paths, "transcript_cleaned_txt")
    raw_transcript = relative_output_path(output_paths, "transcript_txt")
    candidates_ocr = relative_output_path(output_paths, "visual_candidates_ocr_json")
    alignment_file = relative_output_path(output_paths, "visual_alignment_json")
    note_file = relative_output_path(output_paths, "note_path")
    corrections_file = relative_output_path(output_paths, "transcript_corrections_json")
    review_segments_file = relative_output_path(output_paths, "review_segments_json")
    metadata_file = relative_output_path(output_paths, "metadata_json")
    return "\n".join(
        [
            "# AGENTS.md",
            "",
            "## Scope",
            "",
            "当前目录是一个单视频笔记生成工作目录。",
            "唯一任务是基于现有中间产物生成或覆盖 `note.md`，不要运行项目脚本、不要运行测试、不要做仓库级操作。",
            "",
            "## Required Inputs",
            "",
            f"- `{note_prompt}`",
            f"- `{outline_file}`",
            f"- `{blocks_file}`",
            f"- `{cleaned_transcript}` 或 `{raw_transcript}`",
            f"- `{candidates_ocr}`",
            f"- `{alignment_file}`",
            "",
            "## Allowed Writes",
            "",
            f"- `{note_file}`",
            "",
            "## Hard Rules",
            "",
            "- 不要运行 `video_to_notes.py`。",
            "- 不要运行测试。",
            f"- 不要修改 `{raw_transcript}`、`{cleaned_transcript}`、`{corrections_file}`、`{review_segments_file}`、`{candidates_ocr}`、`{outline_file}`、`{blocks_file}`、`{alignment_file}`、`{metadata_file}` 或其他文件。",
            f"- 必须先读取 `{note_prompt}`、`{outline_file}`、`{blocks_file}`。",
            f"- 只允许生成或覆盖 `{note_file}`。",
            "",
        ]
    )


def build_note_retry_prompt(output_paths: dict[str, Path | str], *, failure_reasons: list[str]) -> str:
    lines = [
        build_note_codex_exec_prompt(output_paths),
        "",
        "上一次 note-generation 未通过质量校验，请直接覆盖 `note.md` 并重点解决以下问题：",
    ]
    lines.extend([f"- {reason}" for reason in failure_reasons])
    return "\n".join(lines)


def run_codex_note_generation(
    output_paths: dict[str, Path | str],
    *,
    timeout_seconds: int | None = 300,
) -> None:
    work_dir = Path(output_paths["work_dir"])
    note_path = Path(output_paths["note_path"])
    agents_path = work_dir / "AGENTS.md"
    prompts = [build_note_codex_exec_prompt(output_paths)]
    failure_reasons: list[str] = []
    last_detail = ""
    attempts: list[dict[str, Any]] = []
    original_agents = agents_path.read_text(encoding="utf-8") if agents_path.exists() else None
    agents_path.write_text(build_note_agent_instructions(output_paths), encoding="utf-8")

    try:
        for attempt, prompt in enumerate(prompts, start=1):
            previous_note = note_path.read_text(encoding="utf-8") if note_path.exists() else None
            attempt_record: dict[str, Any] = {
                "attempt": attempt,
                "prompt_type": "initial" if attempt == 1 else "retry",
                "command_status": "started",
                "failure_reasons": [],
                "detail_excerpt": "",
                "quality_gate_passed": False,
                "note_changed": False,
            }
            try:
                result = run_command(
                    [
                        "codex",
                        "exec",
                        "--full-auto",
                        "-c",
                        'model_reasoning_effort="low"',
                        "--skip-git-repo-check",
                        prompt,
                    ],
                    cwd=work_dir,
                    timeout=timeout_seconds,
                )
            except subprocess.CalledProcessError as exc:
                stdout = exc.output.decode("utf-8", errors="ignore") if isinstance(exc.output, bytes) else (exc.output or "")
                stderr = exc.stderr.decode("utf-8", errors="ignore") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
                last_detail = (stdout.strip() or stderr.strip())[:500]
                if note_path.exists():
                    note_path.write_text(
                        sanitize_note_body_timestamps(note_path.read_text(encoding="utf-8")),
                        encoding="utf-8",
                    )
                current_note = note_path.read_text(encoding="utf-8") if note_path.exists() else None
                note_changed = current_note != previous_note
                attempt_record["note_changed"] = note_changed
                failure_reasons = explain_note_generation_failure(note_path)
                if not failure_reasons:
                    failure_reasons = ["codex note-generation 返回非零退出码"]
                if not note_changed:
                    failure_reasons.append("note.md 未被本次 note-generation 更新")
                attempt_record["command_status"] = "nonzero_exit"
                attempt_record["detail_excerpt"] = last_detail
                attempt_record["failure_reasons"] = list(dict.fromkeys(failure_reasons))
            except subprocess.TimeoutExpired as exc:
                stdout = exc.stdout.decode("utf-8", errors="ignore") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
                stderr = exc.stderr.decode("utf-8", errors="ignore") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
                last_detail = (stdout.strip() or stderr.strip())[:500]
                if note_path.exists():
                    note_path.write_text(
                        sanitize_note_body_timestamps(note_path.read_text(encoding="utf-8")),
                        encoding="utf-8",
                    )
                current_note = note_path.read_text(encoding="utf-8") if note_path.exists() else None
                note_changed = current_note != previous_note
                attempt_record["note_changed"] = note_changed
                if note_generation_completed(note_path) and note_changed:
                    attempt_record["command_status"] = "timeout_with_valid_note"
                    attempt_record["detail_excerpt"] = last_detail
                    attempt_record["quality_gate_passed"] = True
                    attempts.append(attempt_record)
                    write_note_generation_report(
                        output_paths,
                        generator="codex",
                        attempts=attempts,
                        status="passed",
                        final_failure_reasons=[],
                    )
                    return
                failure_reasons = explain_note_generation_failure(note_path)
                if not note_changed:
                    failure_reasons.append("note.md 未被本次 note-generation 更新")
                timeout_reason = f"codex note-generation 超时（>{timeout_seconds} 秒）"
                if timeout_reason not in failure_reasons:
                    failure_reasons.append(timeout_reason)
                attempt_record["command_status"] = "timeout"
                attempt_record["detail_excerpt"] = last_detail
                attempt_record["failure_reasons"] = list(dict.fromkeys(failure_reasons))
            else:
                last_detail = result.stdout.strip() or result.stderr.strip()
                if note_path.exists():
                    note_path.write_text(
                        sanitize_note_body_timestamps(note_path.read_text(encoding="utf-8")),
                        encoding="utf-8",
                    )
                current_note = note_path.read_text(encoding="utf-8") if note_path.exists() else None
                note_changed = current_note != previous_note
                attempt_record["note_changed"] = note_changed
                if note_generation_completed(note_path) and note_changed:
                    attempt_record["command_status"] = "completed"
                    attempt_record["detail_excerpt"] = last_detail[:500]
                    attempt_record["quality_gate_passed"] = True
                    attempts.append(attempt_record)
                    write_note_generation_report(
                        output_paths,
                        generator="codex",
                        attempts=attempts,
                        status="passed",
                        final_failure_reasons=[],
                    )
                    return
                failure_reasons = explain_note_generation_failure(note_path)
                if not note_changed:
                    failure_reasons.append("note.md 未被本次 note-generation 更新")
                attempt_record["command_status"] = "quality_gate_failed"
                attempt_record["detail_excerpt"] = last_detail[:500]
                attempt_record["failure_reasons"] = list(dict.fromkeys(failure_reasons))

            attempts.append(attempt_record)

            if attempt == 1:
                prompts.append(build_note_retry_prompt(output_paths, failure_reasons=failure_reasons))
                continue
            break

        write_note_generation_report(
            output_paths,
            generator="codex",
            attempts=attempts,
            status="failed",
            final_failure_reasons=failure_reasons,
        )
        detail = "; ".join(failure_reasons) if failure_reasons else (last_detail or "note-generation failed")
        raise RuntimeError(f"note-generation failed after retry: {detail}")
    finally:
        if original_agents is None:
            agents_path.unlink(missing_ok=True)
        else:
            agents_path.write_text(original_agents, encoding="utf-8")


def run_command(
    command: list[str],
    *,
    cwd: Path | None = None,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        command,
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = process.communicate()
        raise subprocess.TimeoutExpired(command, timeout or 0, output=stdout, stderr=stderr) from exc

    completed = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command, output=stdout, stderr=stderr)
    return completed


def require_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Required binary not found: {name}")


def prepare_codex_review_workspace(output_paths: dict[str, Path | str]) -> Path:
    review_workspace = Path(tempfile.mkdtemp(prefix="codex_review_"))
    files_to_copy = (
        "transcript_txt",
        "transcript_cleaned_txt",
        "transcript_corrections_json",
        "codex_review_prompt_md",
        "work_dir_agents_md",
        "review_segments_json",
        "visual_candidates_ocr_json",
    )

    for key in files_to_copy:
        source = Path(output_paths[key])
        if source.exists():
            shutil.copy2(source, review_workspace / source.name)

    visual_candidates_dir = Path(output_paths["visual_candidates_dir"])
    if visual_candidates_dir.exists():
        shutil.copytree(
            visual_candidates_dir,
            review_workspace / visual_candidates_dir.name,
            dirs_exist_ok=True,
        )

    return review_workspace


def sync_codex_review_outputs(review_workspace: Path, output_paths: dict[str, Path | str]) -> None:
    for key in ("transcript_cleaned_txt", "transcript_corrections_json"):
        target = Path(output_paths[key])
        source = review_workspace / target.name
        if source.exists():
            shutil.copy2(source, target)


def ffprobe_duration(input_video: Path) -> float:
    result = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(input_video),
        ]
    )
    return float(result.stdout.strip())


def extract_audio(input_video: Path, audio_path: Path) -> None:
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_video),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(audio_path),
        ]
    )


def transcribe_audio(audio_path: Path, output_dir: Path, *, model: str, language: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            sys.executable,
            "-m",
            "whisper",
            str(audio_path),
            "--model",
            model,
            "--language",
            language,
            "--task",
            "transcribe",
            "--output_dir",
            str(output_dir),
        ]
    )


def plan_frame_timestamps(
    *,
    duration_seconds: float,
    frame_interval: int,
    max_frames: int,
    transcript_segments: list[dict[str, Any]] | None = None,
) -> list[int]:
    if max_frames <= 0 or duration_seconds <= 0:
        return []

    if transcript_segments:
        bucket_size = duration_seconds / max_frames
        chosen: list[int] = []
        seen: set[int] = set()
        for index in range(max_frames):
            bucket_start = index * bucket_size
            bucket_end = duration_seconds if index == max_frames - 1 else (index + 1) * bucket_size
            bucket_segments = []
            for segment in transcript_segments:
                start = float(segment.get("start", 0.0) or 0.0)
                end = float(segment.get("end", start) or start)
                midpoint = (start + end) / 2
                if bucket_start <= midpoint < bucket_end or (index == max_frames - 1 and midpoint <= bucket_end):
                    bucket_segments.append(segment)

            if bucket_segments:
                best = max(bucket_segments, key=lambda item: len(str(item.get("text", "")).strip()))
                timestamp = int(round((float(best.get("start", 0.0) or 0.0) + float(best.get("end", 0.0) or 0.0)) / 2))
            else:
                timestamp = int(round(bucket_start))

            timestamp = max(0, min(int(duration_seconds), timestamp))
            if timestamp not in seen:
                chosen.append(timestamp)
                seen.add(timestamp)

        if chosen:
            return chosen[:max_frames]

    timestamps = []
    current = 0
    while current < duration_seconds and len(timestamps) < max_frames:
        timestamps.append(current)
        current += frame_interval

    if timestamps and timestamps[-1] != 0 and len(timestamps) < max_frames and duration_seconds > timestamps[-1]:
        midpoint = int(duration_seconds / 2)
        if midpoint not in timestamps:
            timestamps.append(midpoint)

    return sorted(timestamps)[:max_frames]


def plan_visual_supplemental_timestamps(
    *,
    duration_seconds: float,
    candidate_frames: list[dict[str, Any]],
    offset_seconds: int = 12,
) -> list[int]:
    existing = {int(round(float(item.get("timestamp", 0.0) or 0.0))) for item in candidate_frames}
    supplemental: list[int] = []
    for frame in candidate_frames:
        if not bool(frame.get("is_low_value", False)):
            continue
        timestamp = int(round(float(frame.get("timestamp", 0.0) or 0.0)))
        for candidate in (timestamp - offset_seconds, timestamp + offset_seconds):
            bounded = max(0, min(int(duration_seconds), candidate))
            if bounded not in existing and bounded not in supplemental:
                supplemental.append(bounded)
    return sorted(supplemental)


def merge_visual_candidates(
    original_frames: list[dict[str, Any]],
    supplemental_frames: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: dict[float, dict[str, Any]] = {}
    for frame in original_frames:
        merged[float(frame.get("timestamp", 0.0) or 0.0)] = frame
    for frame in supplemental_frames:
        merged[float(frame.get("timestamp", 0.0) or 0.0)] = frame
    return [merged[timestamp] for timestamp in sorted(merged)]


def extract_frames_at_timestamps(
    input_video: Path,
    assets_dir: Path,
    *,
    timestamps: list[int],
) -> list[dict[str, Any]]:
    assets_dir.mkdir(parents=True, exist_ok=True)
    frames: list[dict[str, Any]] = []
    work_dir: Path | None = None
    for parent in assets_dir.parents:
        if parent.name == "pipeline":
            work_dir = parent.parent
            break
    existing_indexes = [
        int(match.group(1))
        for path in assets_dir.glob("frame_*.jpg")
        if (match := re.match(r"frame_(\d+)\.jpg$", path.name))
    ]
    next_index = max(existing_indexes, default=0) + 1

    for timestamp in timestamps:
        frame_name = f"frame_{next_index:03d}.jpg"
        frame_path = assets_dir / frame_name
        run_command(
            [
                "ffmpeg",
                "-y",
                "-ss",
                format_seconds(timestamp),
                "-i",
                str(input_video),
                "-frames:v",
                "1",
                str(frame_path),
            ]
        )
        frames.append(
            {
                "timestamp": float(timestamp),
                "path": str(frame_path),
                "relative_path": (
                    frame_path.relative_to(work_dir).as_posix()
                    if work_dir is not None
                    else f"{assets_dir.name}/{frame_name}"
                ),
            }
        )
        next_index += 1
    return frames


def extract_frames(
    input_video: Path,
    assets_dir: Path,
    *,
    duration_seconds: float,
    frame_interval: int,
    max_frames: int,
    transcript_segments: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    timestamps = plan_frame_timestamps(
        duration_seconds=duration_seconds,
        frame_interval=frame_interval,
        max_frames=max_frames,
        transcript_segments=transcript_segments,
    )
    return extract_frames_at_timestamps(input_video, assets_dir, timestamps=timestamps)


def run_tesseract(image_path: Path) -> str:
    for language in ("chi_sim+eng", "eng"):
        result = subprocess.run(
            ["tesseract", str(image_path), "stdout", "-l", language],
            text=True,
            capture_output=True,
        )
        text = result.stdout.strip()
        if text:
            return " ".join(line.strip() for line in text.splitlines() if line.strip())
    return ""


def ocr_frames(frames: list[dict[str, Any]], ocr_json_path: Path) -> list[dict[str, Any]]:
    results = []
    for frame in frames:
        image_path = Path(frame["path"])
        with tempfile.TemporaryDirectory(prefix="ocr-prep-") as tmp_dir:
            preprocessed_path = Path(tmp_dir) / image_path.name
            preprocess_meta = preprocess_image_for_ocr(image_path, preprocessed_path)
            ocr_text = run_tesseract(preprocessed_path)
        enriched = dict(frame)
        enriched["ocr_text"] = ocr_text
        enriched["ocr_source"] = "preprocessed"
        enriched["ocr_quality_score"] = ocr_quality_score(ocr_text)
        enriched["ocr_preprocess"] = {
            "cropped": bool(preprocess_meta.get("cropped", False)),
            "steps": list(preprocess_meta.get("steps", [])),
        }
        results.append(enriched)

    results = annotate_scene_change_scores(results)
    ocr_json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def _ocr_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[0-9A-Za-z]+|[\u4e00-\u9fff]{2,}", text.lower())
        if len(token.strip()) >= 2
    }


def _informative_ocr_tokens(text: str) -> set[str]:
    weak_tokens = {"transformer", "bert"}
    return {token for token in _ocr_tokens(text) if token not in weak_tokens}


def _segment_ends_with_sentence(text: str) -> bool:
    normalized = str(text).strip()
    return bool(re.search(r"[。！？!?；;.]([\"'”’)\]]+)?$", normalized))


def _frame_hash_signature(frame: dict[str, Any]) -> str:
    explicit = str(frame.get("phash", "")).strip()
    if explicit:
        return explicit
    path = Path(str(frame.get("path", "")).strip())
    if str(frame.get("path", "")).strip() and path.exists() and path.is_file():
        return hashlib.md5(path.read_bytes()).hexdigest()[:16]
    return ""


def _hash_distance(left: str, right: str) -> int:
    if not left or not right or len(left) != len(right):
        return 999
    return sum(1 for a, b in zip(left, right) if a != b)


def _ocr_jaccard(left: str, right: str) -> float:
    left_tokens = _ocr_tokens(left)
    right_tokens = _ocr_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    union = left_tokens | right_tokens
    if not union:
        return 0.0
    return len(left_tokens & right_tokens) / len(union)


def select_representative_frame(cluster: list[dict[str, Any]]) -> dict[str, Any]:
    if not cluster:
        return {}
    center = sum(float(item.get("timestamp", 0.0) or 0.0) for item in cluster) / len(cluster)

    def score(frame: dict[str, Any]) -> tuple[float, float]:
        ocr_text = str(frame.get("ocr_text", "")).strip()
        token_count = len(_ocr_tokens(ocr_text))
        ocr_len = len(ocr_text)
        timestamp = float(frame.get("timestamp", 0.0) or 0.0)
        information_score = float(frame.get("ocr_quality_score", 0.0) or 0.0) + token_count * 10 + float(frame.get("scene_change_score", 0.0) or 0.0)
        middle_distance = abs(timestamp - center)
        return (information_score, -middle_distance)

    return max(cluster, key=score)


def build_visual_units(frames: list[dict[str, Any]]) -> dict[str, Any]:
    if not frames:
        return {"unit_count": 0, "units": []}

    sorted_frames = sorted(frames, key=lambda item: float(item.get("timestamp", 0.0) or 0.0))
    page_change_candidates = [frame for frame in sorted_frames if bool(frame.get("page_change_candidate", False))]
    if page_change_candidates:
        seen_timestamps: set[float] = set()
        filtered_frames: list[dict[str, Any]] = []
        for frame in [sorted_frames[0], *page_change_candidates, sorted_frames[-1]]:
            timestamp = float(frame.get("timestamp", 0.0) or 0.0)
            if timestamp in seen_timestamps:
                continue
            filtered_frames.append(frame)
            seen_timestamps.add(timestamp)
        sorted_frames = filtered_frames
    clusters: list[list[dict[str, Any]]] = []

    for frame in sorted_frames:
        if not clusters:
            clusters.append([frame])
            continue

        previous = clusters[-1][-1]
        hash_distance = _hash_distance(_frame_hash_signature(previous), _frame_hash_signature(frame))
        ocr_similarity = _ocr_jaccard(str(previous.get("ocr_text", "")), str(frame.get("ocr_text", "")))

        if hash_distance <= 6 or ocr_similarity >= 0.7:
            clusters[-1].append(frame)
        else:
            clusters.append([frame])

    units = []
    for index, cluster in enumerate(clusters, start=1):
        representative = select_representative_frame(cluster)
        representative_ocr = str(representative.get("ocr_text", "")).strip()
        representative_tokens = _ocr_tokens(representative_ocr)
        informative_tokens = _informative_ocr_tokens(representative_ocr)
        units.append(
            {
                "unit_id": f"visual_unit_{index:03d}",
                "start": float(cluster[0].get("timestamp", 0.0) or 0.0),
                "end": float(cluster[-1].get("timestamp", 0.0) or 0.0),
                "frame_count": len(cluster),
                "frame_paths": [str(item.get("relative_path", "")).strip() for item in cluster if str(item.get("relative_path", "")).strip()],
                "representative_frame": str(representative.get("relative_path", "")).strip(),
                "representative_timestamp": float(representative.get("timestamp", 0.0) or 0.0),
                "ocr_text": representative_ocr,
                "ocr_len": len(representative_ocr),
                "ocr_quality_score": float(representative.get("ocr_quality_score", 0.0) or 0.0),
                "scene_change_score": float(representative.get("scene_change_score", 0.0) or 0.0),
                "change_ratio": float(representative.get("change_ratio", 0.0) or 0.0),
                "change_kind": str(representative.get("change_kind", "no_change")).strip() or "no_change",
                "is_low_value": (
                    float(representative.get("ocr_quality_score", 0.0) or 0.0) < 25.0
                    or len(representative_ocr) < 20
                    or len(informative_tokens) < 2
                    or (len(representative_tokens) <= 2 and len(representative_ocr) < 30)
                ),
                "selection_reason": [
                    "highest_ocr_density",
                    "scene_change_aware",
                    "sequential_dedup_cluster",
                    "page_change_candidate" if bool(representative.get("page_change_candidate", False)) else "candidate_fallback",
                ],
            }
        )

    return {"unit_count": len(units), "units": units}


def read_excerpt(transcript_txt: Path, *, limit: int = 6) -> list[str]:
    if not transcript_txt.exists():
        return []
    lines = [line.strip() for line in transcript_txt.read_text(encoding="utf-8").splitlines() if line.strip()]
    return lines[:limit]


def preferred_transcript_path(transcript_txt: Path, transcript_cleaned_txt: Path) -> Path:
    if transcript_cleaned_txt.exists() and transcript_cleaned_txt.read_text(encoding="utf-8").strip():
        return transcript_cleaned_txt
    return transcript_txt


def load_transcript_segments(transcript_json: Path) -> list[dict[str, Any]]:
    if not transcript_json.exists():
        return []

    payload = json.loads(transcript_json.read_text(encoding="utf-8"))
    segments = payload.get("segments")
    if not isinstance(segments, list):
        return []

    normalized: list[dict[str, Any]] = []
    for index, segment in enumerate(segments, start=1):
        if not isinstance(segment, dict):
            continue
        text = " ".join(str(segment.get("text", "")).split())
        if not text:
            continue
        start = float(segment.get("start", 0.0) or 0.0)
        end = float(segment.get("end", start) or start)
        if end < start:
            end = start
        normalized.append(
            {
                "index": index,
                "start": start,
                "end": end,
                "text": text,
            }
        )
    return normalized


def build_review_segments(
    *,
    transcript_txt: Path,
    transcript_json: Path,
    frames: list[dict[str, Any]],
    max_chars: int = 280,
    max_duration: float = 150.0,
) -> dict[str, Any]:
    raw_segments = load_transcript_segments(transcript_json)
    if not raw_segments:
        fallback_text = " ".join(
            line.strip() for line in transcript_txt.read_text(encoding="utf-8").splitlines() if line.strip()
        ) if transcript_txt.exists() else ""
        raw_segments = [
            {
                "index": 1,
                "start": 0.0,
                "end": 0.0,
                "text": fallback_text,
            }
        ] if fallback_text else []

    groups: list[list[dict[str, Any]]] = []
    index = 0
    grace_chars = 120
    grace_duration = 30.0
    sentence_extension_limit = 2

    while index < len(raw_segments):
        current_group = [raw_segments[index]]
        current_chars = len(raw_segments[index]["text"])
        current_start = float(raw_segments[index]["start"])
        index += 1

        while index < len(raw_segments):
            segment = raw_segments[index]
            proposed_chars = current_chars + 1 + len(segment["text"])
            proposed_duration = float(segment["end"]) - current_start
            if proposed_chars > max_chars or proposed_duration > max_duration:
                if current_group and not _segment_ends_with_sentence(current_group[-1]["text"]):
                    extension_count = 0
                    while index < len(raw_segments) and extension_count < sentence_extension_limit:
                        extension = raw_segments[index]
                        extension_chars = current_chars + 1 + len(extension["text"])
                        extension_duration = float(extension["end"]) - current_start
                        if extension_chars > max_chars + grace_chars or extension_duration > max_duration + grace_duration:
                            break
                        current_group.append(extension)
                        current_chars = extension_chars
                        index += 1
                        extension_count += 1
                        if _segment_ends_with_sentence(extension["text"]):
                            break
                break

            current_group.append(segment)
            current_chars = proposed_chars
            index += 1

        groups.append(current_group)

    review_segments = []
    for group_index, group in enumerate(groups, start=1):
        start = float(group[0]["start"])
        end = float(group[-1]["end"])
        ocr_hints = []
        for frame in frames:
            timestamp = float(frame.get("timestamp", 0.0) or 0.0)
            if start - 5 <= timestamp <= end + 5:
                text = str(frame.get("ocr_text", "")).strip()
                if text and text not in ocr_hints:
                    ocr_hints.append(text)
            if len(ocr_hints) >= 3:
                break

        text = " ".join(item["text"] for item in group).strip()
        review_segments.append(
            {
                "segment_id": f"segment_{group_index:03d}",
                "start": start,
                "end": end,
                "label": f"{format_seconds(start)}-{format_seconds(end)}",
                "segment_indexes": [int(item["index"]) for item in group],
                "char_count": len(text),
                "text": text,
                "ocr_hints": ocr_hints,
            }
        )

    return {
        "source_transcript": transcript_txt.name,
        "source_transcript_json": transcript_json.name,
        "segment_count": len(review_segments),
        "segments": review_segments,
    }


def load_review_segments(review_segments_json: Path) -> list[dict[str, Any]]:
    if not review_segments_json.exists():
        return []
    payload = json.loads(review_segments_json.read_text(encoding="utf-8"))
    segments = payload.get("segments")
    return segments if isinstance(segments, list) else []


def segment_review_template(review_segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "segment_id": str(segment.get("segment_id", "")),
            "start": float(segment.get("start", 0.0) or 0.0),
            "end": float(segment.get("end", 0.0) or 0.0),
            "summary": "",
            "issues": [],
            "status": "pending",
        }
        for segment in review_segments
    ]


def correction_template(
    transcript_txt: Path,
    transcript_cleaned_txt: Path,
    review_segments: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "source_transcript": transcript_txt.name,
        "cleaned_transcript": transcript_cleaned_txt.name,
        "review_status": "pending",
        "last_updated": None,
        "segment_reviews": segment_review_template(review_segments),
        "corrections": [
            {
                "raw": "",
                "cleaned": "",
                "reason": "",
                "evidence": [],
            }
        ],
    }


def normalize_corrections_payload(
    payload: dict[str, Any],
    transcript_txt: Path,
    transcript_cleaned_txt: Path,
    review_segments: list[dict[str, Any]],
) -> dict[str, Any]:
    template = correction_template(transcript_txt, transcript_cleaned_txt, review_segments)
    normalized = dict(template)
    normalized.update(
        {
            "source_transcript": payload.get("source_transcript", template["source_transcript"]),
            "cleaned_transcript": payload.get("cleaned_transcript", template["cleaned_transcript"]),
            "review_status": payload.get("review_status", template["review_status"]),
            "last_updated": payload.get("last_updated", template["last_updated"]),
        }
    )

    segment_reviews = payload.get("segment_reviews")
    if isinstance(segment_reviews, list) and segment_reviews:
        normalized["segment_reviews"] = []
        for entry in segment_reviews:
            if not isinstance(entry, dict):
                continue
            normalized["segment_reviews"].append(
                {
                    "segment_id": entry.get("segment_id", ""),
                    "start": float(entry.get("start", 0.0) or 0.0),
                    "end": float(entry.get("end", 0.0) or 0.0),
                    "summary": entry.get("summary", ""),
                    "issues": entry.get("issues", [] if entry.get("issues") is None else entry.get("issues")),
                    "status": entry.get("status", "pending"),
                }
            )
        if not normalized["segment_reviews"]:
            normalized["segment_reviews"] = template["segment_reviews"]

    corrections = payload.get("corrections")
    if isinstance(corrections, list) and corrections:
        normalized["corrections"] = []
        for entry in corrections:
            if not isinstance(entry, dict):
                continue
            normalized["corrections"].append(
                {
                    "raw": entry.get("raw", ""),
                    "cleaned": entry.get("cleaned", ""),
                    "reason": entry.get("reason", ""),
                    "evidence": entry.get("evidence", [] if entry.get("evidence") is None else entry.get("evidence")),
                }
            )
        if not normalized["corrections"]:
            normalized["corrections"] = template["corrections"]
    return normalized


def build_codex_review_prompt(
    *,
    source_video: Path,
    transcript_txt: Path,
    transcript_cleaned_txt: Path,
    transcript_corrections_json: Path,
    review_segments_json: Path,
    review_segments: list[dict[str, Any]],
    frames: list[dict[str, Any]],
) -> str:
    lines = [
        "# Codex Review Prompt",
        "",
        "## Goal",
        "",
        "把课程视频的原始 Whisper 转写整理成适合学习的术语准确文本。",
        "",
        "## Inputs",
        "",
        f"- 来源视频：`{source_video.name}`",
        f"- 原始转写：`{transcript_txt.name}`",
        f"- 分段审阅计划：`{review_segments_json.name}`",
        "",
        "## Outputs",
        "",
        f"- 清洗稿：`{transcript_cleaned_txt.name}`",
        f"- 纠错记录：`{transcript_corrections_json.name}`",
        "",
        "只修改 `transcript.cleaned.txt` 和 `transcript.corrections.json`。",
        "",
        "## Steps",
        "",
        "1. 阅读 `transcript.txt`、`review_segments.json` 和 `visual_candidates.ocr.json`，按分段顺序理解讲解内容。",
        "2. 结合 OCR、候选画面线索和课程上下文，逐段修正模型名、术语名、代码关键词。",
        "3. 将修正后的连续文本写入 `transcript.cleaned.txt`。",
        "4. 在 `transcript.corrections.json.segment_reviews` 中为每个分段填写总结、问题点和完成状态。",
        "5. 将关键修正逐条填入 `transcript.corrections.json.corrections`。",
        "6. 不要扩写没有依据的新知识，只做纠错、断句、措辞澄清。",
        "",
        "## Instructions",
        "",
        "- 保留原意，不要扩写无依据内容。",
        "- 优先修正模型名、术语名、代码关键词、公式相关词。",
        "- 不使用静态术语字典，结合上下文和画面 OCR 判断。",
        "- 必须逐段审阅，不要只做全局粗略改写。",
        "- 在 `transcript.cleaned.txt` 中写入校正后的连续文本。",
        "- 在 `transcript.corrections.json` 中记录分段审阅结果和关键纠正项。",
        "",
        "## Corrections JSON Schema",
        "",
        "```json",
        json.dumps(
            {
                "source_transcript": transcript_txt.name,
                "cleaned_transcript": transcript_cleaned_txt.name,
                "review_status": "pending|in_progress|done",
                "last_updated": "ISO-8601 or null",
                "segment_reviews": [
                    {
                        "segment_id": "segment_001",
                        "start": 0.0,
                        "end": 120.0,
                        "summary": "本段主要讲了什么",
                        "issues": ["本段修正了哪些术语或歧义"],
                        "status": "pending|done",
                    }
                ],
                "corrections": [
                    {
                        "raw": "原错误词",
                        "cleaned": "修正后词语",
                        "reason": "为什么这样改",
                        "evidence": ["OCR 线索", "上下文线索"],
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        "```",
        "",
        "## Review Segments",
        "",
    ]

    if review_segments:
        for segment in review_segments:
            lines.append(
                f"- {segment['segment_id']} {segment['label']}: {segment['text']}"
            )
            if segment.get("ocr_hints"):
                lines.append(f"  OCR: {' | '.join(segment['ocr_hints'])}")
    else:
        lines.append("- 无分段信息")

    lines.extend(
        [
            "",
        "## OCR Hints",
        "",
        ]
    )

    if frames:
        for frame in frames[:8]:
            ts = format_seconds(float(frame["timestamp"]))
            ocr_text = frame.get("ocr_text") or "(empty)"
            lines.append(f"- {ts}: {ocr_text}")
    else:
        lines.append("- 无 OCR 线索")

    lines.extend(
        [
            "",
            "## Raw Transcript Preview",
            "",
        ]
    )

    preview_lines = read_excerpt(transcript_txt, limit=12)
    if preview_lines:
        lines.extend([f"- {line}" for line in preview_lines])
    else:
        lines.append("- 原始转写为空")

    lines.extend(
        [
            "",
            "## Acceptance Criteria",
            "",
            "- `transcript.cleaned.txt` 可直接用于学习，模型名和技术术语尽量准确。",
            "- `transcript.txt` 保持不变。",
            "- `segment_reviews` 必须覆盖 `review_segments.json` 中的全部分段，且每段都要有非空总结。",
            "- `transcript.corrections.json` 至少记录主要术语修正，字段完整。",
            "- 如果无法确认的词，保守处理，并在 `reason` 或 `evidence` 中说明不确定性。",
            "",
        ]
    )
    return "\n".join(lines)


def build_codex_exec_prompt(output_paths: dict[str, Path | str]) -> str:
    return "\n".join(
        [
            "请阅读当前目录中的 `codex_review_prompt.md` 并严格执行。",
            "你的任务只包括：",
            "1. 阅读 `transcript.txt`、`review_segments.json`、`visual_candidates.ocr.json`、`visual_candidates/` 和 `codex_review_prompt.md`。",
            "2. 修正 `transcript.cleaned.txt`。",
            "3. 按 `review_segments.json` 逐段填写 `transcript.corrections.json.segment_reviews`。",
            "4. 填写 `transcript.corrections.json.corrections`。",
            "5. 将 `transcript.corrections.json` 中的 `review_status` 更新为 `done`，并填写 `last_updated` 为当前 ISO-8601 时间。",
            "限制：",
            "- 不要修改 transcript.txt。",
            "- 不要修改 note.md、metadata.json、visual_candidates.ocr.json 或其他文件。",
            "- `transcript.cleaned.txt` 不能与 `transcript.txt` 完全相同，至少要完成实质术语纠正和必要的病句整理。",
            "- `segment_reviews` 必须覆盖 `review_segments.json` 中列出的每个 segment_id，且每段 `summary` 非空、`status` 为 `done`。",
            "- `transcript.corrections.json` 至少记录关键纠错项；如果原始转写较长，通常至少要有 3 条有效纠错。",
            "- 每条有效纠错都要有 `raw`、`cleaned`、`reason`，并尽量提供 `evidence`。",
            "- 如果某些词无法完全确认，保守修正，并把不确定性写入 `reason` 或 `evidence`。",
            f"- 只允许改动 `{Path(output_paths['transcript_cleaned_txt']).name}` 和 `{Path(output_paths['transcript_corrections_json']).name}`。",
        ]
    )


def build_codex_retry_prompt(output_paths: dict[str, Path | str], *, failure_reasons: list[str]) -> str:
    lines = [
        build_codex_exec_prompt(output_paths),
        "",
        "上一次审阅结果未通过质量校验，请直接覆盖修正这两个文件，并重点解决以下问题：",
    ]
    lines.extend([f"- {reason}" for reason in failure_reasons])
    return "\n".join(lines)


def needs_json_only_retry(failure_reasons: list[str]) -> bool:
    if not failure_reasons:
        return False

    allowed_prefixes = (
        "review_status 不是 done",
        "last_updated 为空",
        "segment_reviews 未覆盖全部分段",
        "有效纠错条目不足",
    )
    return all(any(reason.startswith(prefix) for prefix in allowed_prefixes) for reason in failure_reasons)


def build_codex_json_retry_prompt(output_paths: dict[str, Path | str], *, failure_reasons: list[str]) -> str:
    transcript_cleaned_txt = Path(output_paths["transcript_cleaned_txt"]).name
    transcript_corrections_json = Path(output_paths["transcript_corrections_json"]).name
    lines = [
        "上一次审阅已经产出了可接受的清洗稿，这次只需要补全 JSON 审阅记录。",
        f"- 不要重写 {transcript_cleaned_txt}。",
        f"- 只需要补全 {transcript_corrections_json}。",
        "- 读取现有的 transcript.cleaned.txt、transcript.txt、review_segments.json、visual_candidates.ocr.json、visual_candidates/ 和 codex_review_prompt.md。",
        "- 将 transcript.corrections.json 中的 review_status 更新为 done，并填写 last_updated。",
        "- segment_reviews 必须覆盖全部 segment_id，每段都要有非空 summary，status 必须为 done。",
        "- corrections 至少补足到质量门要求的有效条目数。",
        "- 不要修改 transcript.txt、review_segments.json、visual_candidates.ocr.json、note.md、metadata.json 或其他文件。",
        "",
        "请重点解决以下剩余问题：",
    ]
    lines.extend([f"- {reason}" for reason in failure_reasons])
    return "\n".join(lines)


def build_retry_prompt(output_paths: dict[str, Path | str], *, failure_reasons: list[str]) -> str:
    if needs_json_only_retry(failure_reasons):
        return build_codex_json_retry_prompt(output_paths, failure_reasons=failure_reasons)
    return build_codex_retry_prompt(output_paths, failure_reasons=failure_reasons)


def build_review_agent_instructions(
    *,
    transcript_txt: Path,
    transcript_cleaned_txt: Path,
    transcript_corrections_json: Path,
    codex_review_prompt_md: Path,
    review_segments_json: Path,
) -> str:
    return "\n".join(
        [
            "# AGENTS.md",
            "",
            "## Scope",
            "",
            "当前目录是一个单视频审阅工作目录，不是项目根目录。",
            "你在这里的唯一任务是完成转写审阅，不要重跑项目脚本，不要执行测试，不要做仓库级操作。",
            "",
            "## Required Inputs",
            "",
            f"- `{codex_review_prompt_md.name}`",
            f"- `{review_segments_json.name}`",
            f"- `{transcript_txt.name}`",
            "- `visual_candidates.ocr.json`",
            "- `visual_candidates/`",
            "",
            "## Allowed Writes",
            "",
            f"- `{transcript_cleaned_txt.name}`",
            f"- `{transcript_corrections_json.name}`",
            "",
            "## Hard Rules",
            "",
            "- 不要运行 `video_to_notes.py`。",
            "- 不要运行测试。",
            "- 不要修改 `transcript.txt`、`note.md`、`metadata.json`、`visual_candidates.ocr.json`、`review_segments.json` 或其他文件。",
            "- 必须先读取 `codex_review_prompt.md` 和 `review_segments.json`，再开始修改。",
            "- 必须按分段填写 `segment_reviews`，并完成关键术语纠错。",
            "- 完成这两个文件后立即结束，不要继续做额外工作。",
            "",
        ]
    )


def related_frames_for_segment(segment: dict[str, Any], frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    start = float(segment.get("start", 0.0) or 0.0)
    end = float(segment.get("end", 0.0) or 0.0)
    selected = []
    for frame in frames:
        timestamp = float(frame.get("timestamp", 0.0) or 0.0)
        if start - 5 <= timestamp <= end + 5:
            selected.append(frame)
    return selected


def build_segment_review_prompt(
    *,
    source_video: Path,
    segment: dict[str, Any],
    transcript_txt: Path,
    transcript_cleaned_txt: Path,
    transcript_corrections_json: Path,
    top_level_prompt: str,
    frames: list[dict[str, Any]],
) -> str:
    lines = [
        "# Segment Codex Review Prompt",
        "",
        "## Goal",
        "",
        "只审阅当前这个 segment，并把结果写入分段输出文件。",
        "",
        "## Inputs",
        "",
        f"- 来源视频：`{source_video.name}`",
        f"- 分段描述：`segment_input.json`",
        f"- 当前分段原稿：`{transcript_txt.name}`",
        "",
        "## Outputs",
        "",
        f"- 当前分段清洗稿：`{transcript_cleaned_txt.name}`",
        f"- 当前分段纠错记录：`{transcript_corrections_json.name}`",
        "",
        "只修改这两个输出文件。",
        "",
        "## Steps",
        "",
        "1. 阅读 `segment_input.json`、`segment_transcript.txt`、`visual_candidates.ocr.json`、`visual_candidates/` 和当前 prompt。",
        "2. 只修正当前 segment 的术语、断句和明显听写错误，不扩写无依据内容。",
        "3. 将修正后的当前 segment 文本写入 `segment.cleaned.txt`。",
        "4. 在 `segment.corrections.json.segment_reviews` 中只填写当前 `segment_id` 的总结、问题点和完成状态。",
        "5. 在 `segment.corrections.json.corrections` 中记录当前 segment 的关键纠错项。",
        "6. 将 `review_status` 设为 `done`，并填写 `last_updated`。",
        "",
        "## Current Segment",
        "",
        f"- segment_id: `{segment.get('segment_id', '')}`",
        f"- time: `{segment.get('label', '')}`",
        f"- text: {segment.get('text', '')}",
    ]

    if segment.get("ocr_hints"):
        lines.append(f"- OCR hints: {' | '.join(str(item) for item in segment['ocr_hints'])}")

    if frames:
        lines.extend(["", "## Nearby OCR", ""])
        for frame in frames:
            lines.append(f"- {format_seconds(float(frame.get('timestamp', 0.0) or 0.0))}: {frame.get('ocr_text') or '(empty)'}")

    lines.extend(
        [
            "",
            "## Acceptance Criteria",
            "",
            "- `segment.cleaned.txt` 不能为空，且不能与 `segment_transcript.txt` 完全相同。",
            "- `segment_reviews` 只能覆盖当前 segment_id，且 summary 非空、status 为 done。",
            "- 每条有效纠错都要有 `raw`、`cleaned`、`reason` 和非空 `evidence`。",
            "",
            "## Global Prompt Context",
            "",
            top_level_prompt.strip(),
            "",
        ]
    )
    return "\n".join(lines)


def build_segment_codex_exec_prompt(segment: dict[str, Any]) -> str:
    return "\n".join(
        [
            "请阅读当前目录中的 `codex_review_prompt.md` 与 `segment_input.json` 并严格执行。",
            "你的任务只包括：",
            "1. 阅读 `segment_transcript.txt`、`segment_input.json`、`visual_candidates.ocr.json`、`visual_candidates/` 和 `codex_review_prompt.md`。",
            "2. 修正 `segment.cleaned.txt`。",
            "3. 在 `segment.corrections.json.segment_reviews` 中填写当前 segment 的总结、问题点和完成状态。",
            "4. 填写 `segment.corrections.json.corrections`。",
            "5. 将 `segment.corrections.json` 中的 `review_status` 更新为 `done`，并填写 `last_updated`。",
            "限制：",
            "- 不要修改 `segment_transcript.txt`、`segment_input.json`、`visual_candidates.ocr.json`、`visual_candidates/` 或其他文件。",
            "- `segment.cleaned.txt` 不能与 `segment_transcript.txt` 完全相同。",
            f"- `segment_reviews` 必须只覆盖 `{segment.get('segment_id', '')}`，且 summary 非空、status 为 done。",
            "- `segment.corrections.json` 至少记录当前 segment 的关键纠错项。",
            "- 每条有效纠错都要有 `raw`、`cleaned`、`reason`，并尽量提供 `evidence`。",
            "- 只允许改动 `segment.cleaned.txt` 和 `segment.corrections.json`。",
        ]
    )


def build_segment_retry_prompt(segment: dict[str, Any], *, failure_reasons: list[str]) -> str:
    lines = [
        build_segment_codex_exec_prompt(segment),
        "",
        "上一次这个 segment 的审阅结果未通过质量校验，请直接覆盖修正这两个文件，并重点解决以下问题：",
    ]
    lines.extend([f"- {reason}" for reason in failure_reasons])
    return "\n".join(lines)


def build_segment_json_retry_prompt(segment: dict[str, Any], *, failure_reasons: list[str]) -> str:
    lines = [
        "上一次这个 segment 的审阅已经产出了可接受的分段清洗稿，这次只需要补全 segment.corrections.json。",
        "- 不要重写 segment.cleaned.txt。",
        "- 只需要补全 segment.corrections.json。",
        "- 读取现有的 segment.cleaned.txt、segment_transcript.txt、segment_input.json、visual_candidates.ocr.json、visual_candidates/ 和 codex_review_prompt.md。",
        "- 将 segment.corrections.json 中的 review_status 更新为 done，并填写 last_updated。",
        f"- segment_reviews 必须只覆盖当前 `{segment.get('segment_id', '')}`，且 summary 非空、status 为 done。",
        "- corrections 至少补足到当前 segment 的质量门要求。",
        "- 不要修改 segment_transcript.txt、segment_input.json、visual_candidates.ocr.json、visual_candidates/ 或其他文件。",
        "",
        "请重点解决以下剩余问题：",
    ]
    lines.extend([f"- {reason}" for reason in failure_reasons])
    return "\n".join(lines)


def build_segment_followup_prompt(segment: dict[str, Any], *, failure_reasons: list[str]) -> str:
    if needs_json_only_retry(failure_reasons):
        return build_segment_json_retry_prompt(segment, failure_reasons=failure_reasons)
    return build_segment_retry_prompt(segment, failure_reasons=failure_reasons)


def build_segment_agent_instructions() -> str:
    return "\n".join(
        [
            "# AGENTS.md",
            "",
            "## Scope",
            "",
            "当前目录是一个单 segment 审阅工作目录。",
            "唯一任务是完成这一段的转写审阅，不要运行项目脚本、不要运行测试、不要做仓库级操作。",
            "",
            "## Required Inputs",
            "",
            "- `codex_review_prompt.md`",
            "- `segment_input.json`",
            "- `segment_transcript.txt`",
            "- `visual_candidates.ocr.json`",
            "- `visual_candidates/`",
            "",
            "## Allowed Writes",
            "",
            "- `segment.cleaned.txt`",
            "- `segment.corrections.json`",
            "",
            "## Hard Rules",
            "",
            "- 不要运行 `video_to_notes.py`。",
            "- 不要运行测试。",
            "- 不要修改 `segment_transcript.txt`、`segment_input.json`、`visual_candidates.ocr.json`、`visual_candidates/` 或其他文件。",
            "- 必须先读取 `codex_review_prompt.md` 和 `segment_input.json`，再开始修改。",
            "- 只能填写当前 segment 的 `segment_reviews`，不要写入其他 segment_id。",
            "- 完成两个输出文件后立即结束。",
            "",
        ]
    )


def prepare_segment_review_workspace(
    *,
    output_paths: dict[str, Path | str],
    source_video: Path,
    segment: dict[str, Any],
    frames: list[dict[str, Any]],
) -> Path:
    workspace = Path(tempfile.mkdtemp(prefix=f"{segment.get('segment_id', 'segment')}_review_"))
    segment_transcript_txt = workspace / "segment_transcript.txt"
    segment_cleaned_txt = workspace / "segment.cleaned.txt"
    segment_corrections_json = workspace / "segment.corrections.json"
    segment_input_json = workspace / "segment_input.json"
    review_prompt_md = workspace / "codex_review_prompt.md"
    agents_md = workspace / "AGENTS.md"
    workspace_visual_candidates = workspace / "visual_candidates"
    workspace_visual_candidates.mkdir(parents=True, exist_ok=True)

    segment_transcript_txt.write_text(str(segment.get("text", "")).strip(), encoding="utf-8")
    segment_cleaned_txt.write_text(str(segment.get("text", "")).strip(), encoding="utf-8")
    segment_input_json.write_text(
        json.dumps(
            {
                "segment": segment,
                "source_video": source_video.name,
                "source_transcript": Path(output_paths["transcript_txt"]).name,
                "global_review_prompt": Path(output_paths["codex_review_prompt_md"]).name,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    segment_corrections_json.write_text(
        json.dumps(
            correction_template(segment_transcript_txt, segment_cleaned_txt, [segment]),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    segment_frames = related_frames_for_segment(segment, frames)
    review_prompt_md.write_text(
        build_segment_review_prompt(
            source_video=source_video,
            segment=segment,
            transcript_txt=segment_transcript_txt,
            transcript_cleaned_txt=segment_cleaned_txt,
            transcript_corrections_json=segment_corrections_json,
            top_level_prompt=Path(output_paths["codex_review_prompt_md"]).read_text(encoding="utf-8"),
            frames=segment_frames,
        ),
        encoding="utf-8",
    )
    agents_md.write_text(build_segment_agent_instructions(), encoding="utf-8")
    (workspace / "visual_candidates.ocr.json").write_text(
        json.dumps(
            [
                {
                    "timestamp": frame.get("timestamp"),
                    "ocr_text": frame.get("ocr_text", ""),
                    "relative_path": frame.get("relative_path", ""),
                }
                for frame in segment_frames
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    work_dir = Path(output_paths["work_dir"])
    for frame in segment_frames:
        relative_path = str(frame.get("relative_path", "")).strip()
        if not relative_path:
            continue
        source = work_dir / relative_path
        if source.exists():
            target = workspace / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

    return workspace

def valid_correction_entries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    corrections = payload.get("corrections")
    if not isinstance(corrections, list):
        return []

    valid_entries = []
    for entry in corrections:
        if not isinstance(entry, dict):
            continue
        if not str(entry.get("raw", "")).strip():
            continue
        if not str(entry.get("cleaned", "")).strip():
            continue
        if not str(entry.get("reason", "")).strip():
            continue
        evidence = entry.get("evidence", [])
        if not isinstance(evidence, list) or not any(str(item).strip() for item in evidence):
            continue
        valid_entries.append(entry)
    return valid_entries


def completed_segment_reviews(
    payload: dict[str, Any],
    review_segments: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    segment_reviews = payload.get("segment_reviews")
    if not isinstance(segment_reviews, list):
        return [], [str(item.get("segment_id", "")) for item in review_segments if str(item.get("segment_id", "")).strip()]

    valid_reviews = []
    covered_ids: set[str] = set()
    for entry in segment_reviews:
        if not isinstance(entry, dict):
            continue
        segment_id = str(entry.get("segment_id", "")).strip()
        summary = str(entry.get("summary", "")).strip()
        status = str(entry.get("status", "")).strip()
        if not segment_id or not summary or status != "done":
            continue
        covered_ids.add(segment_id)
        valid_reviews.append(entry)

    required_ids = [str(item.get("segment_id", "")).strip() for item in review_segments if str(item.get("segment_id", "")).strip()]
    missing_ids = [segment_id for segment_id in required_ids if segment_id not in covered_ids]
    return valid_reviews, missing_ids


def required_correction_count(raw_text: str) -> int:
    return 3 if len(raw_text.strip()) >= 200 else 1


def explain_segment_review_failure(
    segment: dict[str, Any],
    transcript_txt: Path,
    transcript_cleaned_txt: Path,
    transcript_corrections_json: Path,
) -> list[str]:
    reasons: list[str] = []

    if not transcript_txt.exists():
        return ["segment_transcript.txt 不存在"]
    if not transcript_cleaned_txt.exists():
        reasons.append("segment.cleaned.txt 不存在")
        return reasons
    if not transcript_corrections_json.exists():
        reasons.append("segment.corrections.json 不存在")
        return reasons

    raw_text = transcript_txt.read_text(encoding="utf-8").strip()
    cleaned_text = transcript_cleaned_txt.read_text(encoding="utf-8").strip()
    if not cleaned_text:
        reasons.append("segment.cleaned.txt 为空")
    elif cleaned_text == raw_text:
        reasons.append("segment.cleaned.txt 与 segment_transcript.txt 完全相同")

    payload = json.loads(transcript_corrections_json.read_text(encoding="utf-8"))
    if payload.get("review_status") != "done":
        reasons.append("review_status 不是 done")
    if not payload.get("last_updated"):
        reasons.append("last_updated 为空")

    _, missing_segment_ids = completed_segment_reviews(payload, [segment])
    if missing_segment_ids:
        reasons.append(f"segment_reviews 未覆盖全部分段: {', '.join(missing_segment_ids)}")

    valid_entries = valid_correction_entries(payload)
    required_count = required_correction_count(raw_text)
    if len(valid_entries) < required_count:
        reasons.append(f"有效纠错条目不足 {required_count} 条")

    return reasons


def segment_review_completed(
    segment: dict[str, Any],
    transcript_txt: Path,
    transcript_cleaned_txt: Path,
    transcript_corrections_json: Path,
) -> bool:
    return not explain_segment_review_failure(
        segment,
        transcript_txt,
        transcript_cleaned_txt,
        transcript_corrections_json,
    )


def merge_segment_review_results(
    *,
    review_segments: list[dict[str, Any]],
    segment_results: dict[str, dict[str, Any]],
    transcript_txt: Path,
    transcript_cleaned_txt: Path,
) -> tuple[str, dict[str, Any]]:
    merged_payload = correction_template(transcript_txt, transcript_cleaned_txt, review_segments)
    merged_payload["review_status"] = "done"
    merged_payload["last_updated"] = iso_timestamp()
    merged_payload["segment_reviews"] = []
    merged_payload["corrections"] = []

    cleaned_parts: list[str] = []
    for segment in review_segments:
        segment_id = str(segment.get("segment_id", "")).strip()
        result = segment_results[segment_id]
        cleaned_text = str(result.get("cleaned_text", "")).strip()
        if cleaned_text:
            cleaned_parts.append(cleaned_text)

        payload = result.get("payload", {})
        segment_reviews = payload.get("segment_reviews", [])
        if isinstance(segment_reviews, list):
            merged_payload["segment_reviews"].extend(segment_reviews)

        merged_payload["corrections"].extend(valid_correction_entries(payload))

    return "\n".join(cleaned_parts), merged_payload


def explain_review_failure(
    transcript_txt: Path,
    transcript_cleaned_txt: Path,
    transcript_corrections_json: Path,
    review_segments_json: Path,
) -> list[str]:
    reasons: list[str] = []

    if not transcript_txt.exists():
        return ["transcript.txt 不存在"]
    if not transcript_cleaned_txt.exists():
        reasons.append("transcript.cleaned.txt 不存在")
        return reasons
    if not transcript_corrections_json.exists():
        reasons.append("transcript.corrections.json 不存在")
        return reasons

    raw_text = transcript_txt.read_text(encoding="utf-8").strip()
    cleaned_text = transcript_cleaned_txt.read_text(encoding="utf-8").strip()
    if not cleaned_text:
        reasons.append("transcript.cleaned.txt 为空")
    elif cleaned_text == raw_text:
        reasons.append("transcript.cleaned.txt 与 transcript.txt 完全相同")

    payload = json.loads(transcript_corrections_json.read_text(encoding="utf-8"))
    if payload.get("review_status") != "done":
        reasons.append("review_status 不是 done")
    if not payload.get("last_updated"):
        reasons.append("last_updated 为空")

    review_segments = load_review_segments(review_segments_json)
    _, missing_segment_ids = completed_segment_reviews(payload, review_segments)
    if missing_segment_ids:
        reasons.append(f"segment_reviews 未覆盖全部分段: {', '.join(missing_segment_ids)}")

    valid_entries = valid_correction_entries(payload)
    required_count = required_correction_count(raw_text)
    if len(valid_entries) < required_count:
        reasons.append(f"有效纠错条目不足 {required_count} 条")

    return reasons


def review_artifacts_completed(
    transcript_txt: Path,
    transcript_cleaned_txt: Path,
    transcript_corrections_json: Path,
    review_segments_json: Path,
) -> bool:
    return not explain_review_failure(
        transcript_txt,
        transcript_cleaned_txt,
        transcript_corrections_json,
        review_segments_json,
    )


def build_review_report(
    *,
    output_paths: dict[str, Path | str],
    transcript_txt: Path,
    transcript_cleaned_txt: Path,
    transcript_corrections_json: Path,
    review_segments_json: Path,
    attempts: list[dict[str, Any]],
    segment_reports: dict[str, dict[str, Any]] | None,
    status: str,
    final_failure_reasons: list[str],
) -> dict[str, Any]:
    raw_text = transcript_txt.read_text(encoding="utf-8").strip() if transcript_txt.exists() else ""
    cleaned_text = transcript_cleaned_txt.read_text(encoding="utf-8").strip() if transcript_cleaned_txt.exists() else ""
    payload: dict[str, Any] = {}
    if transcript_corrections_json.exists():
        payload = json.loads(transcript_corrections_json.read_text(encoding="utf-8"))

    valid_entries = valid_correction_entries(payload)
    review_segments = load_review_segments(review_segments_json)
    valid_segment_reviews, missing_segment_ids = completed_segment_reviews(payload, review_segments)
    attempt_count = sum(
        int(item.get("attempt_count", 0) or 0) for item in (segment_reports or {}).values()
    ) or len(attempts)
    return {
        "source_transcript": transcript_txt.name,
        "cleaned_transcript": transcript_cleaned_txt.name,
        "corrections_file": transcript_corrections_json.name,
        "review_segments_file": review_segments_json.name,
        "status": status,
        "attempt_count": attempt_count,
        "attempts": attempts,
        "segment_results": segment_reports or {},
        "cleaned_differs_from_raw": bool(raw_text and cleaned_text and cleaned_text != raw_text),
        "planned_segment_count": len(review_segments),
        "reviewed_segment_count": len(valid_segment_reviews),
        "missing_segment_ids": missing_segment_ids,
        "valid_correction_count": len(valid_entries),
        "required_correction_count": required_correction_count(raw_text) if raw_text else 1,
        "review_status": payload.get("review_status"),
        "last_updated": payload.get("last_updated"),
        "final_failure_reasons": final_failure_reasons,
        "work_dir": str(output_paths["work_dir"]),
    }


def write_review_report(
    output_paths: dict[str, Path | str],
    *,
    transcript_txt: Path,
    transcript_cleaned_txt: Path,
    transcript_corrections_json: Path,
    review_segments_json: Path,
    attempts: list[dict[str, Any]],
    segment_reports: dict[str, dict[str, Any]] | None,
    status: str,
    final_failure_reasons: list[str],
) -> None:
    Path(output_paths["review_report_json"]).write_text(
        json.dumps(
            build_review_report(
                output_paths=output_paths,
                transcript_txt=transcript_txt,
                transcript_cleaned_txt=transcript_cleaned_txt,
                transcript_corrections_json=transcript_corrections_json,
                review_segments_json=review_segments_json,
                attempts=attempts,
                segment_reports=segment_reports,
                status=status,
                final_failure_reasons=final_failure_reasons,
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def run_segment_codex_review(
    *,
    output_paths: dict[str, Path | str],
    source_video: Path,
    segment: dict[str, Any],
    frames: list[dict[str, Any]],
    timeout_seconds: int | None,
) -> dict[str, Any]:
    segment_transcript_txt_name = "segment_transcript.txt"
    segment_cleaned_txt_name = "segment.cleaned.txt"
    segment_corrections_json_name = "segment.corrections.json"
    review_workspace = prepare_segment_review_workspace(
        output_paths=output_paths,
        source_video=source_video,
        segment=segment,
        frames=frames,
    )

    segment_transcript_txt = review_workspace / segment_transcript_txt_name
    segment_cleaned_txt = review_workspace / segment_cleaned_txt_name
    segment_corrections_json = review_workspace / segment_corrections_json_name

    prompts = [build_segment_codex_exec_prompt(segment)]
    attempt_reports: list[dict[str, Any]] = []
    failure_reasons: list[str] = []
    last_detail = ""

    try:
        for attempt, prompt in enumerate(prompts, start=1):
            try:
                result = run_command(
                    [
                        "codex",
                        "exec",
                        "--full-auto",
                        "-c",
                        'model_reasoning_effort="low"',
                        "--skip-git-repo-check",
                        prompt,
                    ],
                    cwd=review_workspace,
                    timeout=timeout_seconds,
                )
            except subprocess.CalledProcessError as exc:
                stdout = exc.output.decode("utf-8", errors="ignore") if isinstance(exc.output, bytes) else (exc.output or "")
                stderr = exc.stderr.decode("utf-8", errors="ignore") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
                last_detail = (stdout.strip() or stderr.strip())[:500]
                passed = segment_review_completed(
                    segment,
                    segment_transcript_txt,
                    segment_cleaned_txt,
                    segment_corrections_json,
                )
                failure_reasons = [] if passed else explain_segment_review_failure(
                    segment,
                    segment_transcript_txt,
                    segment_cleaned_txt,
                    segment_corrections_json,
                )
                if not passed and not failure_reasons:
                    failure_reasons = ["codex exec 返回非零退出码"]
                attempt_reports.append(
                    {
                        "segment_id": segment.get("segment_id", ""),
                        "attempt": attempt,
                        "prompt_type": "initial" if attempt == 1 else "retry",
                        "passed": passed,
                        "timed_out": False,
                        "failure_reasons": failure_reasons,
                        "codex_output_excerpt": last_detail,
                    }
                )
                if passed:
                    break
                if attempt == 1:
                    prompts.append(build_segment_followup_prompt(segment, failure_reasons=failure_reasons))
                    continue
                break
            except subprocess.TimeoutExpired as exc:
                timeout_reason = f"codex exec 超时（>{timeout_seconds} 秒）"
                stdout = exc.stdout.decode("utf-8", errors="ignore") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
                stderr = exc.stderr.decode("utf-8", errors="ignore") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
                last_detail = (stdout.strip() or stderr.strip())[:500]
                passed = segment_review_completed(
                    segment,
                    segment_transcript_txt,
                    segment_cleaned_txt,
                    segment_corrections_json,
                )
                failure_reasons = [] if passed else explain_segment_review_failure(
                    segment,
                    segment_transcript_txt,
                    segment_cleaned_txt,
                    segment_corrections_json,
                )
                if not passed and timeout_reason not in failure_reasons:
                    failure_reasons.append(timeout_reason)
                attempt_reports.append(
                    {
                        "segment_id": segment.get("segment_id", ""),
                        "attempt": attempt,
                        "prompt_type": "initial" if attempt == 1 else "retry",
                        "passed": passed,
                        "timed_out": True,
                        "failure_reasons": failure_reasons,
                        "codex_output_excerpt": last_detail,
                    }
                )
                if passed:
                    break
                if attempt == 1:
                    prompts.append(build_segment_followup_prompt(segment, failure_reasons=failure_reasons))
                    continue
                break

            passed = segment_review_completed(
                segment,
                segment_transcript_txt,
                segment_cleaned_txt,
                segment_corrections_json,
            )
            failure_reasons = [] if passed else explain_segment_review_failure(
                segment,
                segment_transcript_txt,
                segment_cleaned_txt,
                segment_corrections_json,
            )
            last_detail = result.stdout.strip() or result.stderr.strip()
            attempt_reports.append(
                {
                    "segment_id": segment.get("segment_id", ""),
                    "attempt": attempt,
                    "prompt_type": "initial" if attempt == 1 else "retry",
                    "passed": passed,
                    "timed_out": False,
                    "failure_reasons": failure_reasons,
                    "codex_output_excerpt": last_detail[:500],
                }
            )

            if passed:
                break

            if attempt == 1:
                prompts.append(build_segment_followup_prompt(segment, failure_reasons=failure_reasons))
    finally:
        cleaned_text = segment_cleaned_txt.read_text(encoding="utf-8").strip() if segment_cleaned_txt.exists() else ""
        payload: dict[str, Any] = {}
        if segment_corrections_json.exists():
            payload = normalize_corrections_payload(
                json.loads(segment_corrections_json.read_text(encoding="utf-8")),
                segment_transcript_txt,
                segment_cleaned_txt,
                [segment],
            )
        shutil.rmtree(review_workspace, ignore_errors=True)

    return {
        "segment_id": str(segment.get("segment_id", "")),
        "passed": not failure_reasons,
        "attempt_count": len(attempt_reports),
        "attempts": attempt_reports,
        "failure_reasons": failure_reasons,
        "last_detail": last_detail,
        "cleaned_text": cleaned_text,
        "payload": payload,
    }


def run_codex_review(
    output_paths: dict[str, Path | str],
    *,
    timeout_seconds: int | None = 300,
    parallelism: int = 2,
) -> None:
    transcript_txt = Path(output_paths["transcript_txt"])
    transcript_cleaned_txt = Path(output_paths["transcript_cleaned_txt"])
    transcript_corrections_json = Path(output_paths["transcript_corrections_json"])
    review_segments_json = Path(output_paths["review_segments_json"])
    review_segments = load_review_segments(review_segments_json)
    source_video = Path(output_paths["work_dir"]) / ".."
    frames: list[dict[str, Any]] = []
    ocr_json = Path(output_paths["visual_candidates_ocr_json"])
    if ocr_json.exists():
        payload = json.loads(ocr_json.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            frames = payload

    if not review_segments:
        failure_reasons = ["review_segments.json 不存在有效分段"]
        write_review_report(
            output_paths,
            transcript_txt=transcript_txt,
            transcript_cleaned_txt=transcript_cleaned_txt,
            transcript_corrections_json=transcript_corrections_json,
            review_segments_json=review_segments_json,
            attempts=[],
            segment_reports={},
            status="failed",
            final_failure_reasons=failure_reasons,
        )
        raise RuntimeError("quality gate failed after retry: review_segments.json 不存在有效分段")

    segment_results: dict[str, dict[str, Any]] = {}
    max_workers = max(1, int(parallelism or 1))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(
                run_segment_codex_review,
                output_paths=output_paths,
                source_video=Path(output_paths["work_dir"]),
                segment=segment,
                frames=frames,
                timeout_seconds=timeout_seconds,
            ): str(segment.get("segment_id", ""))
            for segment in review_segments
        }
        for future in as_completed(future_map):
            segment_id = future_map[future]
            segment_results[segment_id] = future.result()

    flattened_attempts: list[dict[str, Any]] = []
    segment_reports: dict[str, dict[str, Any]] = {}
    segment_level_failure_reasons: list[str] = []

    for segment in review_segments:
        segment_id = str(segment.get("segment_id", ""))
        result = segment_results[segment_id]
        flattened_attempts.extend(result["attempts"])
        segment_reports[segment_id] = {
            "attempt_count": result["attempt_count"],
            "passed": result["passed"],
            "failure_reasons": result["failure_reasons"],
        }
        if not result["passed"]:
            segment_level_failure_reasons.extend([f"{segment_id}: {reason}" for reason in result["failure_reasons"]])

    merged_cleaned_text, merged_payload = merge_segment_review_results(
        review_segments=review_segments,
        segment_results=segment_results,
        transcript_txt=transcript_txt,
        transcript_cleaned_txt=transcript_cleaned_txt,
    )
    transcript_cleaned_txt.write_text(merged_cleaned_text, encoding="utf-8")
    transcript_corrections_json.write_text(
        json.dumps(merged_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    failure_reasons = explain_review_failure(
        transcript_txt,
        transcript_cleaned_txt,
        transcript_corrections_json,
        review_segments_json,
    )
    if failure_reasons:
        final_failure_reasons = segment_level_failure_reasons + [
            reason for reason in failure_reasons if reason not in segment_level_failure_reasons
        ]
        write_review_report(
            output_paths,
            transcript_txt=transcript_txt,
            transcript_cleaned_txt=transcript_cleaned_txt,
            transcript_corrections_json=transcript_corrections_json,
            review_segments_json=review_segments_json,
            attempts=flattened_attempts,
            segment_reports=segment_reports,
            status="failed",
            final_failure_reasons=final_failure_reasons,
        )
        raise RuntimeError(f"quality gate failed after retry: {'; '.join(final_failure_reasons)}")

    write_review_report(
        output_paths,
        transcript_txt=transcript_txt,
        transcript_cleaned_txt=transcript_cleaned_txt,
        transcript_corrections_json=transcript_corrections_json,
        review_segments_json=review_segments_json,
        attempts=flattened_attempts,
        segment_reports=segment_reports,
        status="passed",
        final_failure_reasons=[],
    )


def ensure_review_artifacts(
    *,
    source_video: Path,
    transcript_txt: Path,
    transcript_cleaned_txt: Path,
    transcript_corrections_json: Path,
    codex_review_prompt_md: Path,
    work_dir_agents_md: Path,
    transcript_json: Path,
    review_segments_json: Path,
    frames: list[dict[str, Any]],
) -> None:
    review_segments_payload = build_review_segments(
        transcript_txt=transcript_txt,
        transcript_json=transcript_json,
        frames=frames,
    )
    review_segments_json.write_text(json.dumps(review_segments_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    review_segments = review_segments_payload["segments"]

    if transcript_txt.exists() and not transcript_cleaned_txt.exists():
        transcript_cleaned_txt.write_text(transcript_txt.read_text(encoding="utf-8"), encoding="utf-8")

    if not transcript_corrections_json.exists():
        transcript_corrections_json.write_text(
            json.dumps(
                correction_template(transcript_txt, transcript_cleaned_txt, review_segments),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    else:
        existing_payload = json.loads(transcript_corrections_json.read_text(encoding="utf-8"))
        transcript_corrections_json.write_text(
            json.dumps(
                normalize_corrections_payload(
                    existing_payload,
                    transcript_txt,
                    transcript_cleaned_txt,
                    review_segments,
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    codex_review_prompt_md.write_text(
        build_codex_review_prompt(
            source_video=source_video,
            transcript_txt=transcript_txt,
            transcript_cleaned_txt=transcript_cleaned_txt,
            transcript_corrections_json=transcript_corrections_json,
            review_segments_json=review_segments_json,
            review_segments=review_segments,
            frames=frames,
        ),
        encoding="utf-8",
    )
    work_dir_agents_md.write_text(
        build_review_agent_instructions(
            transcript_txt=transcript_txt,
            transcript_cleaned_txt=transcript_cleaned_txt,
            transcript_corrections_json=transcript_corrections_json,
            codex_review_prompt_md=codex_review_prompt_md,
            review_segments_json=review_segments_json,
        ),
        encoding="utf-8",
    )


def move_whisper_outputs(output_paths: dict[str, Path | str]) -> None:
    work_dir = output_paths["work_dir"]
    audio_stem = Path(output_paths["audio_path"]).stem
    mapping = {
        work_dir / f"{audio_stem}.txt": output_paths["transcript_txt"],
        work_dir / f"{audio_stem}.srt": output_paths["transcript_srt"],
        work_dir / f"{audio_stem}.json": output_paths["transcript_json"],
        work_dir / f"{audio_stem}.vtt": output_paths["transcript_vtt"],
        work_dir / f"{audio_stem}.tsv": output_paths["transcript_tsv"],
    }

    for source, target in mapping.items():
        if source.exists():
            source.replace(target)


def write_metadata(
    *,
    output_paths: dict[str, Path | str],
    input_video: Path,
    duration_seconds: float,
    frame_interval: int,
    max_frames: int,
    whisper_model: str,
    language: str,
    visual_source_mode: str,
    slides_path: Path | None,
    slides_usable: bool,
) -> None:
    metadata = {
        "input_video": str(input_video),
        "slug": output_paths["slug"],
        "duration_seconds": duration_seconds,
        "frame_interval": frame_interval,
        "max_frames": max_frames,
        "whisper_model": whisper_model,
        "language": language,
        "visual_source_mode": visual_source_mode,
        "slides_path": str(slides_path) if slides_path else "",
        "slides_usable": slides_usable,
    }
    Path(output_paths["metadata_json"]).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def ensure_dirs(output_paths: dict[str, Path | str]) -> None:
    for key, value in output_paths.items():
        if key == "slug" or not isinstance(value, Path):
            continue
        if key.endswith("_dir"):
            value.mkdir(parents=True, exist_ok=True)
        else:
            value.parent.mkdir(parents=True, exist_ok=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a first-pass study note from a single video.")
    parser.add_argument("--input", required=True, help="Path to the source mp4 file")
    parser.add_argument("--output-root", default="notes", help="Directory for generated notes")
    parser.add_argument("--frame-interval", type=int, default=60, help="Seconds between sampled frames")
    parser.add_argument("--max-frames", type=int, default=4, help="Maximum sampled frames")
    parser.add_argument("--whisper-model", default="small", help="Whisper model name")
    parser.add_argument("--language", default="zh", help="Whisper language code")
    parser.add_argument("--slides", help="Optional path to a course slide PDF used for slide matching")
    parser.add_argument(
        "--visual-source-mode",
        choices=("auto", "slides-first", "video-first"),
        default="auto",
        help="Choose whether final visuals come primarily from usable slides or video-derived frames",
    )
    parser.add_argument("--codex-timeout-seconds", type=int, default=300, help="Timeout for each codex exec attempt")
    parser.add_argument(
        "--codex-review-parallelism",
        type=int,
        default=5,
        help="Maximum number of segment review codex jobs to run in parallel",
    )
    parser.add_argument("--force-audio", action="store_true", help="Rebuild audio.wav even if it already exists")
    parser.add_argument(
        "--force-transcribe",
        action="store_true",
        help="Re-run Whisper transcription and downstream stages even if transcript outputs already exist",
    )
    parser.add_argument("--force-frames", action="store_true", help="Re-extract frames and downstream stages")
    parser.add_argument("--force-ocr", action="store_true", help="Re-run OCR and downstream stages")
    parser.add_argument(
        "--force-review-artifacts",
        action="store_true",
        help="Rebuild review prompt/materials and rerun downstream stages",
    )
    parser.add_argument("--force-codex-review", action="store_true", help="Re-run Codex review and note generation")
    parser.add_argument("--force-note", action="store_true", help="Rebuild note.md even if it already exists")
    parser.add_argument("--skip-codex-review", action="store_true", help="Skip automatic Codex review step")
    parser.add_argument("--skip-codex-note", action="store_true", help="Skip automatic Codex note-generation stage")
    return parser.parse_args(argv)


def main() -> None:
    from video_to_notes_pipeline import run_pipeline

    args = parse_args()
    result = run_pipeline(args)
    print(f"Generated note at {result['output_paths']['note_path']}")


if __name__ == "__main__":
    main()

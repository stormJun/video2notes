from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageFilter, ImageOps, ImageStat


def _trim_uniform_border(image: Image.Image) -> tuple[Image.Image, bool]:
    background = Image.new(image.mode, image.size, image.getpixel((0, 0)))
    diff = ImageChops.difference(image, background)
    bbox = diff.getbbox()
    if not bbox:
        return image, False
    left, top, right, bottom = bbox
    if left == 0 and top == 0 and right == image.size[0] and bottom == image.size[1]:
        return image, False
    return image.crop(bbox), True


def preprocess_image_for_ocr(image_path: Path, output_path: Path) -> dict[str, Any]:
    with Image.open(image_path) as source:
        grayscale = ImageOps.grayscale(source)
        cropped, cropped_flag = _trim_uniform_border(grayscale)
        enhanced = ImageOps.autocontrast(cropped, cutoff=1)
        enhanced = enhanced.filter(ImageFilter.SHARPEN)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        enhanced.save(output_path)
    return {
        "source_path": str(image_path),
        "output_path": str(output_path),
        "cropped": cropped_flag,
        "steps": ["grayscale", "trim_border", "autocontrast", "sharpen"],
    }


def ocr_quality_score(text: str) -> float:
    normalized = " ".join(text.split()).strip()
    if not normalized:
        return 0.0
    tokens = re.findall(r"[0-9A-Za-z]+|[\u4e00-\u9fff]{2,}", normalized.lower())
    weird_chars = sum(1 for char in normalized if not (char.isalnum() or "\u4e00" <= char <= "\u9fff" or char in " -_/.:,;()[]{}+*=%"))
    return float(len(normalized) + len(tokens) * 8 - weird_chars * 2)


def _ocr_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[0-9A-Za-z]+|[\u4e00-\u9fff]{2,}", text.lower())
        if len(token.strip()) >= 2
    }


def _ocr_similarity(left: str, right: str) -> float:
    left_tokens = _ocr_tokens(left)
    right_tokens = _ocr_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    union = left_tokens | right_tokens
    if not union:
        return 0.0
    return len(left_tokens & right_tokens) / len(union)


def _analysis_gray_image(image: np.ndarray | None) -> np.ndarray | None:
    if image is None:
        return None
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(image, (320, 180))
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    height, width = blurred.shape
    # Ignore fixed chrome regions that often contain player UI or presenter overlays.
    blurred[0 : int(height * 0.06), :] = 0
    blurred[int(height * 0.90) :, :] = 0
    blurred[int(height * 0.72) :, int(width * 0.78) :] = 0
    return blurred


def _analysis_gray(path: Path) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return _analysis_gray_image(image)


def _change_metrics(previous: np.ndarray, current: np.ndarray, *, pixel_threshold: int = 20) -> dict[str, float]:
    diff = cv2.absdiff(previous, current)
    scene_score = float(np.mean(diff))
    _, mask = cv2.threshold(diff, pixel_threshold, 255, cv2.THRESH_BINARY)
    changed_pixels = int(np.count_nonzero(mask))
    total_pixels = int(mask.shape[0] * mask.shape[1]) if mask.size else 1
    change_ratio = changed_pixels / total_pixels

    largest_component_ratio = 0.0
    component_count = 0
    if changed_pixels:
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        component_areas = [int(area) for area in stats[1:, cv2.CC_STAT_AREA] if int(area) > 0]
        component_count = len(component_areas)
        if component_areas:
            largest_component_ratio = max(component_areas) / total_pixels

    return {
        "scene_score": scene_score,
        "change_ratio": float(change_ratio),
        "largest_component_ratio": float(largest_component_ratio),
        "component_count": float(component_count),
    }


def annotate_scene_change_scores(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sorted_frames = sorted(frames, key=lambda item: float(item.get("timestamp", 0.0) or 0.0))
    annotated: list[dict[str, Any]] = []
    previous_gray: np.ndarray | None = None
    recent_significant_changes: list[bool] = []

    min_change_ratio = 0.02
    annotation_upper_ratio = 0.08
    strong_page_change_ratio = 0.12
    largest_component_annotation_ratio = 0.02
    stable_window = 5

    for frame in sorted_frames:
        enriched = dict(frame)
        scene_score = 0.0
        change_ratio = 0.0
        largest_component_ratio = 0.0
        component_count = 0.0
        ocr_similarity = 0.0
        change_kind = "no_change"
        page_change_candidate = False
        path = Path(str(frame.get("path", "")).strip())
        if path.exists():
            analysis_gray = _analysis_gray(path)
            if analysis_gray is not None:
                if previous_gray is not None:
                    metrics = _change_metrics(previous_gray, analysis_gray)
                    scene_score = float(metrics["scene_score"])
                    change_ratio = float(metrics["change_ratio"])
                    largest_component_ratio = float(metrics["largest_component_ratio"])
                    component_count = float(metrics["component_count"])
                    ocr_similarity = _ocr_similarity(
                        str(annotated[-1].get("ocr_text", "")).strip() if annotated else "",
                        str(frame.get("ocr_text", "")).strip(),
                    )
                    significant_change = change_ratio >= min_change_ratio
                    broad_change = (
                        change_ratio >= strong_page_change_ratio
                        or (
                            change_ratio >= annotation_upper_ratio
                            and largest_component_ratio >= largest_component_annotation_ratio
                        )
                    )
                    previous_window = recent_significant_changes[-stable_window:]
                    previous_frames_stable = not any(previous_window) if previous_window else True

                    if significant_change:
                        change_kind = "annotation_like_change"
                    if broad_change and previous_frames_stable and ocr_similarity < 0.82:
                        change_kind = "page_change"
                        if annotated:
                            annotated[-1]["page_change_candidate"] = True
                            annotated[-1]["page_change_reason"] = "stable_before_change"
                    recent_significant_changes.append(significant_change)
                    if len(recent_significant_changes) > stable_window:
                        recent_significant_changes.pop(0)
                previous_gray = analysis_gray
        enriched["scene_change_score"] = scene_score
        enriched["change_ratio"] = change_ratio
        enriched["largest_component_ratio"] = largest_component_ratio
        enriched["change_component_count"] = component_count
        enriched["ocr_similarity_to_previous"] = ocr_similarity
        enriched["change_kind"] = change_kind
        enriched["page_change_candidate"] = page_change_candidate
        annotated.append(enriched)

    if annotated:
        annotated[0]["page_change_candidate"] = True
        annotated[0]["page_change_reason"] = annotated[0].get("page_change_reason", "opening_frame")
        annotated[-1]["page_change_candidate"] = True
        annotated[-1]["page_change_reason"] = annotated[-1].get("page_change_reason", "closing_frame")

    return annotated


def scan_visual_candidate_timestamps(
    input_video: Path,
    *,
    duration_seconds: float,
    max_candidates: int,
    sample_interval_seconds: float = 1.0,
    min_gap_seconds: int = 8,
) -> list[int]:
    if max_candidates <= 0 or duration_seconds <= 0:
        return []

    capture = cv2.VideoCapture(str(input_video))
    if not capture.isOpened():
        return [0]

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0.0 or frame_count <= 0:
        capture.release()
        return [0]

    sample_step = max(1, int(round(fps * sample_interval_seconds)))
    min_change_ratio = 0.02
    annotation_upper_ratio = 0.08
    strong_page_change_ratio = 0.12
    largest_component_annotation_ratio = 0.02
    stable_window = 5

    previous_gray: np.ndarray | None = None
    previous_timestamp = 0.0
    recent_significant_changes: list[bool] = []
    raw_candidates: list[dict[str, float]] = [{"timestamp": 0.0, "score": 0.0}]

    try:
        for frame_index in range(0, frame_count, sample_step):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                continue
            timestamp = frame_index / fps
            analysis_gray = _analysis_gray_image(frame)
            if analysis_gray is None:
                continue
            if previous_gray is not None:
                metrics = _change_metrics(previous_gray, analysis_gray)
                change_ratio = float(metrics["change_ratio"])
                largest_component_ratio = float(metrics["largest_component_ratio"])
                significant_change = change_ratio >= min_change_ratio
                broad_change = (
                    change_ratio >= strong_page_change_ratio
                    or (
                        change_ratio >= annotation_upper_ratio
                        and largest_component_ratio >= largest_component_annotation_ratio
                    )
                )
                previous_frames_stable = not any(recent_significant_changes[-stable_window:]) if recent_significant_changes else True
                if broad_change and previous_frames_stable:
                    raw_candidates.append(
                        {
                            "timestamp": previous_timestamp,
                            "score": float(
                                metrics["scene_score"]
                                + change_ratio * 100.0
                                + largest_component_ratio * 50.0
                            ),
                        }
                    )
                recent_significant_changes.append(significant_change)
                if len(recent_significant_changes) > stable_window:
                    recent_significant_changes.pop(0)
            previous_gray = analysis_gray
            previous_timestamp = timestamp
    finally:
        capture.release()

    closing_timestamp = max(0.0, duration_seconds - max(sample_interval_seconds, 1.0))
    raw_candidates.append({"timestamp": closing_timestamp, "score": 0.0})

    deduped_by_timestamp: dict[int, float] = {}
    for candidate in raw_candidates:
        timestamp = max(0, min(int(round(candidate["timestamp"])), int(duration_seconds)))
        deduped_by_timestamp[timestamp] = max(float(candidate["score"]), deduped_by_timestamp.get(timestamp, float("-inf")))

    sorted_candidates = [
        {"timestamp": float(timestamp), "score": score}
        for timestamp, score in sorted(deduped_by_timestamp.items())
    ]

    filtered: list[dict[str, float]] = []
    for candidate in sorted_candidates:
        if not filtered:
            filtered.append(candidate)
            continue
        distance = abs(candidate["timestamp"] - filtered[-1]["timestamp"])
        if distance < min_gap_seconds:
            if candidate["score"] >= filtered[-1]["score"]:
                filtered[-1] = candidate
            continue
        filtered.append(candidate)

    if len(filtered) <= max_candidates:
        return [int(item["timestamp"]) for item in sorted(filtered, key=lambda item: item["timestamp"])]

    strongest = sorted(filtered, key=lambda item: (-item["score"], item["timestamp"]))[:max_candidates]
    return sorted({int(item["timestamp"]) for item in strongest}) or [0]

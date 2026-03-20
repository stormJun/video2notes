from __future__ import annotations

import json
from pathlib import Path
from typing import Any


STAGE_ORDER = (
    "audio",
    "transcribe",
    "frames",
    "ocr",
    "review_artifacts",
    "codex_review",
    "note",
    "metadata",
)


PIPELINE_STAGE_DIR_KEYS = (
    "pipeline_run_dir",
    "pipeline_media_dir",
    "pipeline_media_audio_dir",
    "pipeline_media_visual_dir",
    "pipeline_review_dir",
    "pipeline_alignment_dir",
    "pipeline_structure_dir",
    "pipeline_note_dir",
)


def _path(output_paths: dict[str, Path | str], key: str) -> Path:
    return Path(output_paths[key])


def _non_empty_file(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def _pipeline_readme_text() -> str:
    return "\n".join(
        [
            "# Pipeline Artifacts",
            "",
            "按主流程阶段整理的 canonical 产物目录：",
            "",
            "- `00_run/`: 运行级汇总，例如 `metadata.json`",
            "- `01_media/audio/`: 音频与转写产物",
            "- `01_media/visual/`: 候选画面、OCR 与视觉单元",
            "- `02_review/`: 分段审阅、清洗稿、纠错与审阅报告",
            "- `03_alignment/`: 图文对齐结果、课件页索引与调试文件",
            "- `04_structure/`: 讲义目录与知识块结构",
            "- `05_note/`: 最终讲义结构、生成 prompt、报告与 `note.md`",
            "",
        ]
    )


def sync_pipeline_artifact_view(output_paths: dict[str, Path | str]) -> None:
    pipeline_dir = _path(output_paths, "pipeline_dir")
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    _path(output_paths, "pipeline_readme_md").write_text(_pipeline_readme_text(), encoding="utf-8")
    for stage_dir_key in PIPELINE_STAGE_DIR_KEYS:
        _path(output_paths, stage_dir_key).mkdir(parents=True, exist_ok=True)


def audio_artifact_ready(audio_path: Path) -> bool:
    return _non_empty_file(audio_path)


def transcript_artifacts_ready(output_paths: dict[str, Path | str]) -> bool:
    required = (
        "transcript_txt",
        "transcript_json",
        "transcript_srt",
        "transcript_vtt",
        "transcript_tsv",
    )
    if any(not _non_empty_file(_path(output_paths, key)) for key in required):
        return False

    try:
        payload = json.loads(_path(output_paths, "transcript_json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    segments = payload.get("segments")
    return isinstance(segments, list) and bool(segments)


def expected_frame_count(*, duration_seconds: float, frame_interval: int, max_frames: int) -> int:
    if max_frames <= 0 or duration_seconds <= 0:
        return 0

    timestamps: list[int] = []
    current = 0
    while current < duration_seconds and len(timestamps) < max_frames:
        timestamps.append(current)
        current += frame_interval

    if timestamps and timestamps[-1] != 0 and len(timestamps) < max_frames and duration_seconds > timestamps[-1]:
        midpoint = int(duration_seconds / 2)
        if midpoint not in timestamps:
            timestamps.append(midpoint)

    return len(sorted(timestamps)[:max_frames])


def frames_artifacts_ready(
    output_paths: dict[str, Path | str],
    *,
    duration_seconds: float,
    frame_interval: int,
    max_frames: int,
) -> bool:
    visual_candidates_dir = _path(output_paths, "visual_candidates_dir")
    if not visual_candidates_dir.exists():
        return False
    expected = expected_frame_count(
        duration_seconds=duration_seconds,
        frame_interval=frame_interval,
        max_frames=max_frames,
    )
    actual = len(list(visual_candidates_dir.glob("frame_*.jpg")))
    return actual >= expected


def ocr_artifacts_ready(output_paths: dict[str, Path | str]) -> bool:
    ocr_json = _path(output_paths, "visual_candidates_ocr_json")
    visual_candidates_dir = _path(output_paths, "visual_candidates_dir")
    if not _non_empty_file(ocr_json) or not visual_candidates_dir.exists():
        return False

    frame_count = len(list(visual_candidates_dir.glob("frame_*.jpg")))
    try:
        payload = json.loads(ocr_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    return isinstance(payload, list) and len(payload) == frame_count


def review_artifacts_ready(output_paths: dict[str, Path | str]) -> bool:
    required = (
        "transcript_cleaned_txt",
        "transcript_corrections_json",
        "codex_review_prompt_md",
        "review_segments_json",
    )
    if any(not _non_empty_file(_path(output_paths, key)) for key in required):
        return False

    try:
        payload = json.loads(_path(output_paths, "transcript_corrections_json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    return all(field in payload for field in ("review_status", "last_updated", "segment_reviews", "corrections"))


def codex_review_ready(output_paths: dict[str, Path | str]) -> bool:
    if not review_artifacts_ready(output_paths):
        return False

    try:
        report = json.loads(_path(output_paths, "review_report_json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    if report.get("status") != "passed":
        return False

    import video_to_notes

    return video_to_notes.review_artifacts_completed(
        _path(output_paths, "transcript_txt"),
        _path(output_paths, "transcript_cleaned_txt"),
        _path(output_paths, "transcript_corrections_json"),
        _path(output_paths, "review_segments_json"),
    )


def note_artifact_ready(output_paths: dict[str, Path | str]) -> bool:
    note_path = _path(output_paths, "note_path")
    report_path = _path(output_paths, "note_generation_report_json")
    if not _non_empty_file(note_path) or not _non_empty_file(report_path):
        return False

    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    return payload.get("status") == "passed" and bool(payload.get("quality_gate_passed"))


def resolve_stage_plan(
    *,
    output_paths: dict[str, Path | str],
    duration_seconds: float,
    frame_interval: int,
    max_frames: int,
    force: dict[str, bool],
) -> dict[str, dict[str, Any]]:
    ready = {
        "audio": audio_artifact_ready(_path(output_paths, "audio_path")),
        "transcribe": transcript_artifacts_ready(output_paths),
        "frames": frames_artifacts_ready(
            output_paths,
            duration_seconds=duration_seconds,
            frame_interval=frame_interval,
            max_frames=max_frames,
        ),
        "ocr": ocr_artifacts_ready(output_paths),
        "review_artifacts": review_artifacts_ready(output_paths),
        "codex_review": codex_review_ready(output_paths),
        "note": note_artifact_ready(output_paths),
        "metadata": False,
    }

    note_only_rerender = bool(force.get("note", False)) and not any(
        bool(force.get(stage, False)) for stage in STAGE_ORDER if stage not in {"note", "metadata"}
    )

    invalidated = False
    stage_plan: dict[str, dict[str, Any]] = {}
    for stage in STAGE_ORDER:
        stage_force = bool(force.get(stage, False))
        if stage == "metadata":
            run = True
            reason = "metadata is always refreshed"
        elif note_only_rerender and stage != "note" and not stage_force and not invalidated and ready[stage]:
            run = False
            reason = "preserved for note-only rerender"
        else:
            run = stage_force or invalidated or not ready[stage]
            if stage_force:
                reason = f"forced rerun for {stage}"
            elif invalidated:
                reason = "rerun because an upstream stage is being recomputed"
            elif not ready[stage]:
                reason = "artifacts missing or incomplete"
            else:
                reason = "artifacts already complete"

        stage_plan[stage] = {"run": run, "reason": reason}
        if stage != "metadata" and run:
            invalidated = True

    return stage_plan

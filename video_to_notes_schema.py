from __future__ import annotations

from datetime import datetime
import re
from pathlib import Path


def slugify_stem(path_str: str) -> str:
    stem = Path(path_str).stem.lower()
    stem = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "-", stem)
    stem = re.sub(r"-{2,}", "-", stem)
    return stem.strip("-") or "video"


def format_seconds(seconds: float) -> str:
    total = max(0, int(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def iso_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def plan_output_paths(input_video: Path, output_root: Path) -> dict[str, Path | str]:
    slug = slugify_stem(str(input_video))
    work_dir = output_root / slug
    pipeline_dir = work_dir / "pipeline"
    pipeline_run_dir = pipeline_dir / "00_run"
    pipeline_media_dir = pipeline_dir / "01_media"
    pipeline_media_audio_dir = pipeline_media_dir / "audio"
    pipeline_media_visual_dir = pipeline_media_dir / "visual"
    pipeline_review_dir = pipeline_dir / "02_review"
    pipeline_alignment_dir = pipeline_dir / "03_alignment"
    pipeline_structure_dir = pipeline_dir / "04_structure"
    pipeline_note_dir = pipeline_dir / "05_note"
    transcript_base = pipeline_media_audio_dir / "transcript"
    return {
        "slug": slug,
        "work_dir": work_dir,
        "pipeline_dir": pipeline_dir,
        "pipeline_run_dir": pipeline_run_dir,
        "pipeline_media_dir": pipeline_media_dir,
        "pipeline_media_audio_dir": pipeline_media_audio_dir,
        "pipeline_media_visual_dir": pipeline_media_visual_dir,
        "pipeline_review_dir": pipeline_review_dir,
        "pipeline_alignment_dir": pipeline_alignment_dir,
        "pipeline_structure_dir": pipeline_structure_dir,
        "pipeline_note_dir": pipeline_note_dir,
        "pipeline_readme_md": pipeline_dir / "README.md",
        "slides_preview_dir": pipeline_alignment_dir / "slides_preview",
        "slides_index_raw_json": pipeline_alignment_dir / "slides_index.raw.json",
        "slides_index_json": pipeline_alignment_dir / "slides_index.json",
        "slides_cleanup_prompt_md": pipeline_alignment_dir / "slides_cleanup_prompt.md",
        "slides_cleanup_report_json": pipeline_alignment_dir / "slides_cleanup_report.json",
        "visual_candidates_dir": pipeline_media_visual_dir / "visual_candidates",
        "audio_path": pipeline_media_audio_dir / "audio.wav",
        "transcript_txt": transcript_base.with_suffix(".txt"),
        "transcript_cleaned_txt": pipeline_review_dir / "transcript.cleaned.txt",
        "transcript_srt": transcript_base.with_suffix(".srt"),
        "transcript_json": transcript_base.with_suffix(".json"),
        "transcript_vtt": transcript_base.with_suffix(".vtt"),
        "transcript_tsv": transcript_base.with_suffix(".tsv"),
        "transcript_corrections_json": pipeline_review_dir / "transcript.corrections.json",
        "codex_review_prompt_md": pipeline_review_dir / "codex_review_prompt.md",
        "work_dir_agents_md": pipeline_run_dir / "AGENTS.md",
        "review_report_json": pipeline_review_dir / "review_report.json",
        "review_segments_json": pipeline_review_dir / "review_segments.json",
        "visual_units_json": pipeline_media_visual_dir / "visual_units.json",
        "visual_alignment_json": pipeline_alignment_dir / "visual_alignment.json",
        "ppt_alignment_debug_md": pipeline_alignment_dir / "ppt_alignment_debug.md",
        "note_outline_json": pipeline_structure_dir / "note_outline.json",
        "note_blocks_json": pipeline_structure_dir / "note_blocks.json",
        "note_generation_prompt_md": pipeline_note_dir / "note_generation_prompt.md",
        "note_generation_report_json": pipeline_note_dir / "note_generation_report.json",
        "visual_candidates_ocr_json": pipeline_media_visual_dir / "visual_candidates.ocr.json",
        "metadata_json": pipeline_run_dir / "metadata.json",
        "note_path": pipeline_note_dir / "note.md",
    }

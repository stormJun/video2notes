import json
from pathlib import Path
import pytest
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import video_to_notes
import video_to_notes_note
import video_to_notes_pipeline
from video_to_notes_slides import (
    assess_slides_payload_for_video,
    determine_visual_source_mode,
    parse_slides_pdf,
    resolve_slides_path,
    sanitize_slides_payload,
    slides_payload_has_noise,
)
from video_to_notes_visual import annotate_scene_change_scores, preprocess_image_for_ocr, scan_visual_candidate_timestamps
from video_to_notes_artifacts import (
    audio_artifact_ready,
    codex_review_ready,
    expected_frame_count,
    frames_artifacts_ready,
    note_artifact_ready,
    ocr_artifacts_ready,
    resolve_stage_plan,
    review_artifacts_ready,
    sync_pipeline_artifact_view,
    transcript_artifacts_ready,
)
from video_to_notes import (
    build_review_segments,
    build_visual_alignment,
    build_note_blocks,
    build_note_generation_prompt,
    build_note_outline,
    build_slides_cleanup_prompt,
    render_ppt_alignment_debug_markdown,
    merge_visual_candidates,
    build_codex_exec_prompt,
    build_codex_retry_prompt,
    build_note_markdown,
    build_visual_units,
    ensure_review_artifacts,
    explain_review_failure,
    format_seconds,
    plan_frame_timestamps,
    plan_visual_supplemental_timestamps,
    parse_args,
    plan_output_paths,
    preferred_transcript_path,
    review_artifacts_completed,
    run_codex_note_generation,
    run_codex_slides_cleanup,
    run_codex_review,
    select_representative_frame,
    slugify_stem,
)


VALID_RULE_BASED_NOTE = "# 讲义\n\n## 知识小结\n\n核心定义卡片\n\n知识框架\n\n[00:58]\n"


def write_review_segments(path: Path, segment_count: int = 1) -> None:
    segments = []
    for index in range(segment_count):
        segments.append(
            {
                "segment_id": f"segment_{index + 1:03d}",
                "start": float(index * 60),
                "end": float((index + 1) * 60),
                "label": f"00:0{index}:00-00:0{index + 1}:00",
                "segment_indexes": [index + 1],
                "char_count": 10,
                "text": f"segment {index + 1}",
                "ocr_hints": [f"hint {index + 1}"],
            }
        )
    path.write_text(
        json.dumps(
            {
                "source_transcript": "transcript.txt",
                "source_transcript_json": "transcript.json",
                "segment_count": segment_count,
                "segments": segments,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def write_segment_review_outputs(
    workspace: Path,
    *,
    cleaned_text: str,
    review_status: str = "done",
    issues: list[str] | None = None,
    corrections: list[dict[str, object]] | None = None,
) -> None:
    segment_input = json.loads((workspace / "segment_input.json").read_text(encoding="utf-8"))
    segment = segment_input["segment"]
    (workspace / "segment.cleaned.txt").write_text(cleaned_text, encoding="utf-8")
    (workspace / "segment.corrections.json").write_text(
        json.dumps(
            {
                "source_transcript": "transcript.txt",
                "cleaned_transcript": "transcript.cleaned.txt",
                "review_status": review_status,
                "last_updated": "2026-03-13T04:00:00+08:00" if review_status == "done" else None,
                "segment_reviews": [
                    {
                        "segment_id": segment["segment_id"],
                        "start": segment["start"],
                        "end": segment["end"],
                        "summary": f"{segment['segment_id']} summary" if review_status == "done" else "",
                        "issues": issues or [],
                        "status": "done" if review_status == "done" else "pending",
                    }
                ],
                "corrections": corrections or [{"raw": "", "cleaned": "", "reason": "", "evidence": []}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_slugify_stem_normalizes_filename() -> None:
    path = "/tmp/4.1BERT-2训练.mp4"
    assert slugify_stem(path) == "4-1bert-2训练"


def test_format_seconds_returns_hh_mm_ss() -> None:
    assert format_seconds(0) == "00:00:00"
    assert format_seconds(12.4) == "00:00:12"
    assert format_seconds(450.049) == "00:07:30"


def test_reader_facing_points_keeps_minimal_cleaning_and_deduplicates() -> None:
    points = video_to_notes_note._reader_facing_points(
        [
            "这里面会有 input id，就是它每一个 token 会对应一个 input id",
            "这里面会有 input id，就是它每一个 token 会对应一个 input id",
            "这个 id 其实对应的就是一个词向量、词嵌入向量",
        ],
        limit=6,
    )

    assert points == [
        "这里面会有 input id，就是它每一个 token 会对应一个 input id",
        "这个 id 其实对应的就是一个词向量、词嵌入向量",
    ]


def test_sanitize_note_body_timestamps_keeps_section_titles_and_strips_body_points() -> None:
    markdown = "\n".join(
        [
            "## 课程讲义",
            "",
            "- 一、BERT [00:00]",
            "  - 1. `CLS` 标记的任务含义 [01:03]",
            "    - 1）定义",
            "      - `CLS` 承担分类汇聚职责 [01:03]",
            "      - 它是预留的读出位 [01:03]",
            "    - 配图与公式",
            "      - ![01:03](../01_media/visual/visual_candidates/frame_004.jpg)",
            "",
            "## 关键截图索引",
        ]
    )

    sanitized = video_to_notes_note.sanitize_note_body_timestamps(markdown)

    assert "- 一、BERT [00:00]" in sanitized
    assert "  - 1. `CLS` 标记的任务含义 [01:03]" in sanitized
    assert "      - `CLS` 承担分类汇聚职责" in sanitized
    assert "      - 它是预留的读出位" in sanitized
    assert "承担分类汇聚职责 [01:03]" not in sanitized
    assert "![01:03](../01_media/visual/visual_candidates/frame_004.jpg)" in sanitized


def test_sanitize_note_body_timestamps_normalizes_pipeline_visual_paths() -> None:
    markdown = "\n".join(
        [
            "## 课程讲义",
            "",
            "  - 1. `CLS` 标记的任务含义 [01:03]",
            "    - 配图与公式",
            "      - ![[01:03]](pipeline/01_media/visual/visual_candidates/frame_004.jpg)",
            "",
            "## 关键截图索引",
            "",
            "![01:03](pipeline/01_media/visual/visual_candidates/frame_004.jpg)",
        ]
    )

    sanitized = video_to_notes_note.sanitize_note_body_timestamps(markdown)

    assert "](pipeline/01_media/visual/visual_candidates/frame_004.jpg)" not in sanitized
    assert "![[01:03]](../01_media/visual/visual_candidates/frame_004.jpg)" in sanitized
    assert "![01:03](../01_media/visual/visual_candidates/frame_004.jpg)" in sanitized


def test_plan_output_paths_builds_expected_layout(tmp_path: Path) -> None:
    result = plan_output_paths(
        input_video=Path("/tmp/4.1BERT-2训练.mp4"),
        output_root=tmp_path,
    )

    assert result["slug"] == "4-1bert-2训练"
    assert result["work_dir"] == tmp_path / "4-1bert-2训练"
    assert result["pipeline_dir"] == tmp_path / "4-1bert-2训练" / "pipeline"
    assert result["pipeline_media_audio_dir"] == tmp_path / "4-1bert-2训练" / "pipeline" / "01_media" / "audio"
    assert result["pipeline_media_visual_dir"] == tmp_path / "4-1bert-2训练" / "pipeline" / "01_media" / "visual"
    assert result["pipeline_review_dir"] == tmp_path / "4-1bert-2训练" / "pipeline" / "02_review"
    assert result["pipeline_alignment_dir"] == tmp_path / "4-1bert-2训练" / "pipeline" / "03_alignment"
    assert result["pipeline_structure_dir"] == tmp_path / "4-1bert-2训练" / "pipeline" / "04_structure"
    assert result["pipeline_note_dir"] == tmp_path / "4-1bert-2训练" / "pipeline" / "05_note"
    assert result["slides_preview_dir"] == tmp_path / "4-1bert-2训练" / "pipeline" / "03_alignment" / "slides_preview"
    assert result["slides_index_raw_json"] == tmp_path / "4-1bert-2训练" / "pipeline" / "03_alignment" / "slides_index.raw.json"
    assert result["slides_index_json"] == tmp_path / "4-1bert-2训练" / "pipeline" / "03_alignment" / "slides_index.json"
    assert result["slides_cleanup_prompt_md"] == tmp_path / "4-1bert-2训练" / "pipeline" / "03_alignment" / "slides_cleanup_prompt.md"
    assert result["slides_cleanup_report_json"] == tmp_path / "4-1bert-2训练" / "pipeline" / "03_alignment" / "slides_cleanup_report.json"
    assert result["visual_candidates_dir"] == tmp_path / "4-1bert-2训练" / "pipeline" / "01_media" / "visual" / "visual_candidates"
    assert result["note_path"] == tmp_path / "4-1bert-2训练" / "pipeline" / "05_note" / "note.md"
    assert result["transcript_txt"] == tmp_path / "4-1bert-2训练" / "pipeline" / "01_media" / "audio" / "transcript.txt"
    assert result["transcript_cleaned_txt"] == tmp_path / "4-1bert-2训练" / "pipeline" / "02_review" / "transcript.cleaned.txt"
    assert result["transcript_srt"] == tmp_path / "4-1bert-2训练" / "pipeline" / "01_media" / "audio" / "transcript.srt"
    assert result["visual_candidates_ocr_json"] == tmp_path / "4-1bert-2训练" / "pipeline" / "01_media" / "visual" / "visual_candidates.ocr.json"
    assert result["metadata_json"] == tmp_path / "4-1bert-2训练" / "pipeline" / "00_run" / "metadata.json"
    assert result["transcript_corrections_json"] == tmp_path / "4-1bert-2训练" / "pipeline" / "02_review" / "transcript.corrections.json"
    assert result["codex_review_prompt_md"] == tmp_path / "4-1bert-2训练" / "pipeline" / "02_review" / "codex_review_prompt.md"
    assert result["work_dir_agents_md"] == tmp_path / "4-1bert-2训练" / "pipeline" / "00_run" / "AGENTS.md"
    assert result["review_report_json"] == tmp_path / "4-1bert-2训练" / "pipeline" / "02_review" / "review_report.json"
    assert result["review_segments_json"] == tmp_path / "4-1bert-2训练" / "pipeline" / "02_review" / "review_segments.json"
    assert result["visual_units_json"] == tmp_path / "4-1bert-2训练" / "pipeline" / "01_media" / "visual" / "visual_units.json"
    assert result["visual_alignment_json"] == tmp_path / "4-1bert-2训练" / "pipeline" / "03_alignment" / "visual_alignment.json"
    assert result["ppt_alignment_debug_md"] == tmp_path / "4-1bert-2训练" / "pipeline" / "03_alignment" / "ppt_alignment_debug.md"
    assert result["note_outline_json"] == tmp_path / "4-1bert-2训练" / "pipeline" / "04_structure" / "note_outline.json"
    assert result["note_blocks_json"] == tmp_path / "4-1bert-2训练" / "pipeline" / "04_structure" / "note_blocks.json"
    assert result["note_generation_prompt_md"] == tmp_path / "4-1bert-2训练" / "pipeline" / "05_note" / "note_generation_prompt.md"
    assert result["note_generation_report_json"] == tmp_path / "4-1bert-2训练" / "pipeline" / "05_note" / "note_generation_report.json"
    assert result["transcript_vtt"] == tmp_path / "4-1bert-2训练" / "pipeline" / "01_media" / "audio" / "transcript.vtt"
    assert result["transcript_tsv"] == tmp_path / "4-1bert-2训练" / "pipeline" / "01_media" / "audio" / "transcript.tsv"


def test_sync_pipeline_artifact_view_ensures_stage_directories(tmp_path: Path) -> None:
    output_paths = plan_output_paths(Path("/tmp/4.1BERT-2训练.mp4"), tmp_path)
    video_to_notes.ensure_dirs(output_paths)
    Path(output_paths["audio_path"]).write_bytes(b"wav")
    Path(output_paths["transcript_txt"]).write_text("hello", encoding="utf-8")
    (Path(output_paths["visual_candidates_dir"]) / "frame_001.jpg").write_bytes(b"jpg")
    Path(output_paths["visual_candidates_ocr_json"]).write_text("[]", encoding="utf-8")
    Path(output_paths["review_segments_json"]).write_text("{}", encoding="utf-8")
    Path(output_paths["visual_alignment_json"]).write_text("{}", encoding="utf-8")
    Path(output_paths["note_path"]).write_text("# note", encoding="utf-8")
    Path(output_paths["metadata_json"]).write_text("{}", encoding="utf-8")

    sync_pipeline_artifact_view(output_paths)

    pipeline_readme = Path(output_paths["pipeline_readme_md"])
    assert pipeline_readme.exists()
    assert Path(output_paths["pipeline_media_audio_dir"]).exists()
    assert Path(output_paths["pipeline_media_visual_dir"]).exists()
    assert Path(output_paths["pipeline_review_dir"]).exists()
    assert Path(output_paths["pipeline_alignment_dir"]).exists()
    assert Path(output_paths["pipeline_structure_dir"]).exists()
    assert Path(output_paths["pipeline_note_dir"]).exists()
    assert Path(output_paths["audio_path"]).exists()
    assert Path(output_paths["visual_candidates_dir"]).exists()
    assert Path(output_paths["review_segments_json"]).exists()
    assert Path(output_paths["note_path"]).exists()


def test_note_artifact_ready_requires_passed_generation_report(tmp_path: Path) -> None:
    output_paths = plan_output_paths(Path("/tmp/4.1BERT-2训练.mp4"), tmp_path)
    video_to_notes.ensure_dirs(output_paths)
    Path(output_paths["note_path"]).write_text("# note\n\n## 知识小结\n\n核心定义卡片\n知识框架\n[00:58]", encoding="utf-8")

    assert note_artifact_ready(output_paths) is False

    Path(output_paths["note_generation_report_json"]).write_text(
        json.dumps({"status": "failed", "quality_gate_passed": False}),
        encoding="utf-8",
    )
    assert note_artifact_ready(output_paths) is False

    Path(output_paths["note_generation_report_json"]).write_text(
        json.dumps({"status": "passed", "quality_gate_passed": True}),
        encoding="utf-8",
    )
    assert note_artifact_ready(output_paths) is True


def test_build_note_markdown_builds_structured_study_note() -> None:
    markdown = build_note_markdown(
        title="4.1BERT-2训练",
        source_video="4.1BERT-2训练.mp4",
        duration_seconds=450.049,
        transcript_excerpt=[
            "首先我们先来看我们这两个句子",
            "第1个句子叫做天气真好",
        ],
        cleaned_segments=[
            "这是清洗后的学习表达 1。",
            "这是清洗后的学习表达 2。",
        ],
        review_segments=[
            {
                "segment_id": "segment_001",
                "start": 0.0,
                "end": 60.0,
                "label": "00:00:00-00:01:00",
                "text": "原始讲解片段 1",
            },
            {
                "segment_id": "segment_002",
                "start": 60.0,
                "end": 120.0,
                "label": "00:01:00-00:02:00",
                "text": "原始讲解片段 2",
            },
        ],
        segment_reviews=[
            {
                "segment_id": "segment_001",
                "start": 0.0,
                "end": 60.0,
                "summary": "讲解 BERT 预训练里的 CLS 与句子组织方式。",
                "issues": ["修正 Bort -> BERT", "修正 下居预测 -> 下句预测"],
                "status": "done",
            },
            {
                "segment_id": "segment_002",
                "start": 60.0,
                "end": 120.0,
                "summary": "说明 CLS 在分类任务中的作用。",
                "issues": ["补全 classification 术语"],
                "status": "done",
            },
        ],
        corrections=[
            {
                "raw": "Bort",
                "cleaned": "BERT",
                "reason": "模型名误听",
                "evidence": ["课件标题", "上下文一致"],
            },
            {
                "raw": "下居预测",
                "cleaned": "下句预测",
                "reason": "BERT 预训练术语",
                "evidence": ["课件术语"],
            },
        ],
        frames=[
            {"timestamp": 12.0, "relative_path": "assets/frame_001.jpg", "ocr_text": "BERT 训练"},
            {"timestamp": 72.0, "relative_path": "assets/frame_002.jpg", "ocr_text": ""},
        ],
    )

    assert "# 4.1BERT-2训练 学习讲义" in markdown
    assert "> 来源视频：`4.1BERT-2训练.mp4`" in markdown
    assert "视频时长：`00:07:30`" in markdown
    assert "## 课程目录" in markdown
    assert "### 一、BERT-2训练 00:00:00" in markdown
    assert "1. BERT 预训练里的 CLS 与句子组织方式 [00:00]" in markdown
    assert "## 课程讲义" in markdown
    assert "#### 1. BERT 预训练里的 CLS 与句子组织方式 [00:00]" in markdown
    assert "1）核心定义卡片" in markdown
    assert "2）知识框架" in markdown
    assert "3）讲解展开" in markdown
    assert "1）核心定义卡片 [00:00]" not in markdown
    assert "2）知识框架 [00:00]" not in markdown
    assert "这是清洗后的学习表达 1" in markdown
    assert "原始讲解片段 1" not in markdown
    assert "术语修正" not in markdown
    assert "`Bort` -> `BERT`" not in markdown
    assert "## 关键截图索引" in markdown
    assert "![00:00:12](assets/frame_001.jpg)" in markdown
    assert "OCR：BERT 训练" not in markdown
    assert "![00:01:12](assets/frame_002.jpg)" in markdown


def test_build_note_markdown_prefers_frames_relevant_to_segment_content() -> None:
    markdown = build_note_markdown(
        title="BERT",
        source_video="bert.mp4",
        duration_seconds=120.0,
        transcript_excerpt=[],
        cleaned_segments=["这里讲 CLS 和 BERT 的输入表示。"],
        review_segments=[
            {
                "segment_id": "segment_001",
                "start": 0.0,
                "end": 60.0,
                "label": "00:00:00-00:01:00",
                "text": "原始片段",
                "ocr_hints": ["BERT CLS"],
            }
        ],
        segment_reviews=[
            {
                "segment_id": "segment_001",
                "start": 0.0,
                "end": 60.0,
                "summary": "本段讲 BERT 输入里的 CLS 标记。",
                "issues": ["修正 Bort -> BERT", "补全 CLS"],
                "status": "done",
            }
        ],
        corrections=[],
        frames=[
            {"timestamp": 12.0, "relative_path": "assets/frame_001.jpg", "ocr_text": "BERT CLS embedding"},
            {"timestamp": 18.0, "relative_path": "assets/frame_002.jpg", "ocr_text": "cat dog holiday"},
            {"timestamp": 50.0, "relative_path": "assets/frame_003.jpg", "ocr_text": ""},
        ],
    )

    structured_notes = markdown.split("## 关键截图索引", 1)[0]
    assert "assets/frame_001.jpg" in structured_notes
    assert "assets/frame_002.jpg" not in structured_notes


def test_build_note_markdown_prefers_visual_units_over_raw_frames_for_structured_notes() -> None:
    markdown = build_note_markdown(
        title="BERT",
        source_video="bert.mp4",
        duration_seconds=120.0,
        transcript_excerpt=[],
        cleaned_segments=["这里讲 CLS 和 BERT 的输入表示。"],
        review_segments=[
            {
                "segment_id": "segment_001",
                "start": 0.0,
                "end": 60.0,
                "label": "00:00:00-00:01:00",
                "text": "原始片段",
                "ocr_hints": ["BERT CLS"],
            }
        ],
        segment_reviews=[
            {
                "segment_id": "segment_001",
                "start": 0.0,
                "end": 60.0,
                "summary": "本段讲 BERT 输入里的 CLS 标记。",
                "issues": ["修正 Bort -> BERT", "补全 CLS"],
                "status": "done",
            }
        ],
        corrections=[],
        visual_units=[
            {
                "unit_id": "visual_unit_001",
                "start": 10.0,
                "end": 20.0,
                "representative_frame": "assets/unit_frame_001.jpg",
                "representative_timestamp": 12.0,
                "ocr_text": "BERT CLS embedding",
                "ocr_len": 18,
                "is_low_value": False,
            }
        ],
        frames=[
            {"timestamp": 12.0, "relative_path": "assets/frame_001.jpg", "ocr_text": "irrelevant raw frame"},
        ],
    )

    structured_notes = markdown.split("## 关键截图索引", 1)[0]
    assert "assets/unit_frame_001.jpg" in structured_notes
    assert "assets/frame_001.jpg" not in structured_notes


def test_build_visual_alignment_prefers_matching_ppt_slide_over_visual_unit() -> None:
    payload = build_visual_alignment(
        review_segments=[
            {
                "segment_id": "segment_001",
                "start": 0.0,
                "end": 60.0,
                "label": "00:00:00-00:01:00",
                "text": "这里讲传统神经网络和 RNN 的区别",
                "ocr_hints": [],
            }
        ],
        segment_reviews=[
            {
                "segment_id": "segment_001",
                "start": 0.0,
                "end": 60.0,
                "summary": "讲解传统神经网络与 RNN 的区别",
                "issues": [],
                "status": "done",
            }
        ],
        cleaned_segments=["传统神经网络不具备时间记忆能力，RNN 引入隐藏状态。"],
        slides=[
            {
                "slide_id": "slide_004",
                "slide_index": 4,
                "title": "传统神经网络",
                "text": "传统神经网络 RNN 对比 隐藏状态",
                "relative_path": "slides_preview/rendered/slide_004.png",
                "is_low_value": False,
            }
        ],
        visual_units=[
            {
                "unit_id": "visual_unit_001",
                "start": 20.0,
                "end": 25.0,
                "representative_frame": "visual_candidates/frame_001.jpg",
                "representative_timestamp": 22.0,
                "ocr_text": "generic architecture",
                "ocr_len": 20,
                "is_low_value": False,
            }
        ],
        frames=[],
    )

    assert payload["visual_source_mode"] == "slides-first"
    segment = payload["segments"][0]
    assert segment["selection_mode"] == "ppt_slide_match"
    assert segment["selected_visuals"][0]["source"] == "ppt_slide"
    assert segment["selected_visuals"][0]["relative_path"] == "slides_preview/rendered/slide_004.png"


def test_build_visual_alignment_prefers_title_matched_early_slide_over_generic_late_body_match() -> None:
    payload = build_visual_alignment(
        review_segments=[
            {
                "segment_id": "segment_001",
                "start": 0.0,
                "end": 60.0,
                "label": "00:00:00-00:01:00",
                "text": "这里讲传统神经网络与 RNN 的区别",
                "ocr_hints": [],
            }
        ],
        segment_reviews=[
            {
                "segment_id": "segment_001",
                "start": 0.0,
                "end": 60.0,
                "summary": "讲解传统神经网络与 RNN 的区别",
                "issues": [],
                "status": "done",
            }
        ],
        cleaned_segments=["传统神经网络不具备时间记忆能力，RNN 会引入隐藏状态。"],
        slides=[
            {
                "slide_id": "slide_004",
                "slide_index": 4,
                "title": "传统神经网络",
                "text": "传统神经网络 DNN",
                "relative_path": "slides_preview/slide_004.png",
                "is_low_value": False,
            },
            {
                "slide_id": "slide_022",
                "slide_index": 22,
                "title": "Slide 22",
                "text": "RNN 隐藏状态 编码器 解码器 RNN 输入 输出 历史信息",
                "relative_path": "slides_preview/slide_022.png",
                "is_low_value": False,
            },
        ],
        visual_units=[],
        frames=[],
    )

    assert payload["segments"][0]["selected_visuals"][0]["slide_index"] == 4
    assert "opening_sequence_preference" in payload["segments"][0]["selected_visuals"][0]["selection_reason"]
    assert "global_sequence_optimization" in payload["segments"][0]["selected_visuals"][0]["selection_reason"]


def test_build_visual_alignment_prefers_forward_ppt_slide_progression() -> None:
    payload = build_visual_alignment(
        review_segments=[
            {
                "segment_id": "segment_001",
                "start": 0.0,
                "end": 60.0,
                "label": "00:00:00-00:01:00",
                "text": "这里讲 RNN 的基本概念",
                "ocr_hints": [],
            },
            {
                "segment_id": "segment_002",
                "start": 60.0,
                "end": 120.0,
                "label": "00:01:00-00:02:00",
                "text": "这里继续讲 RNN 的计算展开",
                "ocr_hints": [],
            },
        ],
        segment_reviews=[
            {
                "segment_id": "segment_001",
                "start": 0.0,
                "end": 60.0,
                "summary": "讲解 RNN 的基本概念",
                "issues": [],
                "status": "done",
            },
            {
                "segment_id": "segment_002",
                "start": 60.0,
                "end": 120.0,
                "summary": "讲解 RNN 的计算展开",
                "issues": [],
                "status": "done",
            },
        ],
        cleaned_segments=[
            "RNN 引入隐藏状态保留时序信息。",
            "RNN 在时间展开后逐步计算隐藏状态。",
        ],
        slides=[
            {
                "slide_id": "slide_001",
                "slide_index": 1,
                "title": "RNN 基本概念",
                "text": "RNN 隐藏状态 当前输入 时序信息",
                "relative_path": "slides_preview/slide_001.png",
                "is_low_value": False,
            },
            {
                "slide_id": "slide_002",
                "slide_index": 2,
                "title": "RNN 计算展开",
                "text": "RNN 隐藏状态 当前输入 时序信息",
                "relative_path": "slides_preview/slide_002.png",
                "is_low_value": False,
            },
        ],
        visual_units=[],
        frames=[],
    )

    assert payload["segments"][0]["selected_visuals"][0]["slide_index"] == 1
    assert payload["segments"][1]["selected_visuals"][0]["slide_index"] == 2
    assert "sequence_advance_bonus" in payload["segments"][1]["selected_visuals"][0]["selection_reason"]
    assert "global_sequence_optimization" in payload["segments"][1]["selected_visuals"][0]["selection_reason"]


def test_build_visual_alignment_uses_global_ppt_sequence_to_avoid_far_jump_then_backtrack() -> None:
    payload = build_visual_alignment(
        review_segments=[
            {
                "segment_id": "segment_001",
                "start": 0.0,
                "end": 60.0,
                "label": "00:00:00-00:01:00",
                "text": "这里先讲 RNN 的基本概念",
                "ocr_hints": [],
            },
            {
                "segment_id": "segment_002",
                "start": 60.0,
                "end": 120.0,
                "label": "00:01:00-00:02:00",
                "text": "这里讲 RNN 在时间维度上的展开计算",
                "ocr_hints": [],
            },
            {
                "segment_id": "segment_003",
                "start": 120.0,
                "end": 180.0,
                "label": "00:02:00-00:03:00",
                "text": "这里讲 RNN 的输出层与结果解释",
                "ocr_hints": [],
            },
        ],
        segment_reviews=[
            {
                "segment_id": "segment_001",
                "start": 0.0,
                "end": 60.0,
                "summary": "讲解 RNN 的基本概念",
                "issues": [],
                "status": "done",
            },
            {
                "segment_id": "segment_002",
                "start": 60.0,
                "end": 120.0,
                "summary": "讲解 RNN 在时间维度上的展开计算",
                "issues": [],
                "status": "done",
            },
            {
                "segment_id": "segment_003",
                "start": 120.0,
                "end": 180.0,
                "summary": "讲解 RNN 的输出层与结果解释",
                "issues": [],
                "status": "done",
            },
        ],
        cleaned_segments=[
            "RNN 通过隐藏状态保留时序信息。",
            "RNN 在时间维度展开后逐步计算隐藏状态，当前输入和上一状态共同参与计算。",
            "输出层根据隐藏状态得到最终结果。",
        ],
        slides=[
            {
                "slide_id": "slide_001",
                "slide_index": 1,
                "title": "RNN 基本概念",
                "text": "RNN 隐藏状态 时序信息",
                "relative_path": "slides_preview/slide_001.png",
                "is_low_value": False,
            },
            {
                "slide_id": "slide_002",
                "slide_index": 2,
                "title": "时间维度展开",
                "text": "时间维度 展开 逐步计算 隐藏状态",
                "relative_path": "slides_preview/slide_002.png",
                "is_low_value": False,
            },
            {
                "slide_id": "slide_003",
                "slide_index": 3,
                "title": "输出层",
                "text": "输出层 最终结果",
                "relative_path": "slides_preview/slide_003.png",
                "is_low_value": False,
            },
            {
                "slide_id": "slide_009",
                "slide_index": 9,
                "title": "RNN 综合结构",
                "text": "RNN 时间维度 展开 逐步计算 隐藏状态 当前输入 上一状态 输出层 最终结果",
                "relative_path": "slides_preview/slide_009.png",
                "is_low_value": False,
            },
        ],
        visual_units=[],
        frames=[],
        visual_source_mode="slides-first",
    )

    selected_indices = [segment["selected_visuals"][0]["slide_index"] for segment in payload["segments"]]
    assert selected_indices == [1, 2, 3]
    assert "global_sequence_optimization" in payload["segments"][1]["selected_visuals"][0]["selection_reason"]
    assert payload["segments"][1]["ppt_sequence_trace"]["selected_slide_index"] == 2
    assert payload["segments"][1]["ppt_sequence_trace"]["previous_slide_index"] == 1
    assert payload["segments"][1]["ppt_sequence_trace"]["sequence_reason"] == "sequence_advance_bonus"
    assert payload["segments"][1]["ppt_sequence_trace"]["global_path_score"] >= payload["segments"][1]["ppt_sequence_trace"]["cumulative_score"]
    losing_candidate = next(
        candidate
        for candidate in payload["segments"][1]["candidate_visuals"]
        if candidate["slide_index"] == 9
    )
    assert losing_candidate["sequence_selected"] is False
    assert losing_candidate["sequence_rejection_reason"] == "lost_to_slide_002_on_global_path_score"
    assert losing_candidate["sequence_score_gap"] > 0


def test_build_note_markdown_prefers_nearby_visual_units_before_far_stronger_match() -> None:
    markdown = build_note_markdown(
        title="BERT",
        source_video="bert.mp4",
        duration_seconds=200.0,
        transcript_excerpt=[],
        cleaned_segments=["这里在讲 CLS 和 SEP。"],
        review_segments=[
            {
                "segment_id": "segment_001",
                "start": 0.0,
                "end": 60.0,
                "label": "00:00:00-00:01:00",
                "text": "原始片段",
                "ocr_hints": ["CLS SEP"],
            }
        ],
        segment_reviews=[
            {
                "segment_id": "segment_001",
                "start": 0.0,
                "end": 60.0,
                "summary": "本段讲 CLS 和 SEP 的输入标记。",
                "issues": ["补全 CLS", "补全 SEP"],
                "status": "done",
            }
        ],
        corrections=[],
        visual_units=[
            {
                "unit_id": "visual_unit_001",
                "start": 20.0,
                "end": 25.0,
                "representative_frame": "assets/near.jpg",
                "representative_timestamp": 22.0,
                "ocr_text": "CLS SEP",
                "ocr_len": 7,
                "is_low_value": False,
            },
            {
                "unit_id": "visual_unit_002",
                "start": 180.0,
                "end": 185.0,
                "representative_frame": "assets/far.jpg",
                "representative_timestamp": 182.0,
                "ocr_text": "BERT CLS SEP embedding token_type position attention softmax",
                "ocr_len": 60,
                "is_low_value": False,
            },
        ],
        frames=[],
    )

    structured_notes = markdown.split("## 关键截图索引", 1)[0]
    assert "assets/near.jpg" in structured_notes
    assert "assets/far.jpg" not in structured_notes


def test_build_note_markdown_filters_low_value_nearby_visual_units() -> None:
    markdown = build_note_markdown(
        title="BERT",
        source_video="bert.mp4",
        duration_seconds=200.0,
        transcript_excerpt=[],
        cleaned_segments=["这里在讲 embedding 和 token_type。"],
        review_segments=[
            {
                "segment_id": "segment_001",
                "start": 120.0,
                "end": 180.0,
                "label": "00:02:00-00:03:00",
                "text": "原始片段",
                "ocr_hints": ["embedding token_type"],
            }
        ],
        segment_reviews=[
            {
                "segment_id": "segment_001",
                "start": 120.0,
                "end": 180.0,
                "summary": "本段讲 embedding 和 token_type。",
                "issues": ["补全 embedding", "补全 token_type"],
                "status": "done",
            }
        ],
        corrections=[],
        visual_units=[
            {
                "unit_id": "visual_unit_001",
                "start": 130.0,
                "end": 132.0,
                "representative_frame": "assets/weak.jpg",
                "representative_timestamp": 131.0,
                "ocr_text": "WF",
                "ocr_len": 2,
                "is_low_value": True,
            },
            {
                "unit_id": "visual_unit_002",
                "start": 145.0,
                "end": 148.0,
                "representative_frame": "assets/strong.jpg",
                "representative_timestamp": 146.0,
                "ocr_text": "embedding token_type position",
                "ocr_len": 29,
                "is_low_value": False,
            },
        ],
        frames=[],
    )

    structured_notes = markdown.split("## 关键截图索引", 1)[0]
    assert "assets/strong.jpg" in structured_notes
    assert "assets/weak.jpg" not in structured_notes


def test_render_ppt_alignment_debug_markdown_includes_selected_and_rejected_candidates() -> None:
    markdown = render_ppt_alignment_debug_markdown(
        title="RNN基本原理",
        visual_alignment={
            "visual_source_mode": "slides-first",
            "segment_count": 1,
            "ppt_slide_count": 3,
            "segments": [
                {
                    "segment_id": "segment_001",
                    "label": "00:00:00-00:01:00",
                    "summary": "讲解 RNN 的基本概念",
                    "selection_mode": "ppt_slide_match",
                    "ppt_sequence_trace": {
                        "selected_slide_index": 2,
                        "base_score": 18.0,
                        "sequence_step_score": 10.0,
                        "cumulative_score": 38.5,
                        "global_path_score": 42.0,
                        "previous_slide_index": 1,
                        "sequence_reason": "sequence_advance_bonus",
                    },
                    "selected_visuals": [
                        {
                            "source": "ppt_slide",
                            "slide_index": 2,
                            "base_score": 18.0,
                            "sequence_score": 10.0,
                            "cumulative_score": 38.5,
                            "global_path_score": 42.0,
                            "sequence_reason": "sequence_advance_bonus",
                            "relative_path": "slides_preview/slide_002.png",
                        }
                    ],
                    "candidate_visuals": [
                        {
                            "source": "ppt_slide",
                            "slide_index": 2,
                            "sequence_selected": True,
                            "base_score": 18.0,
                            "sequence_score": 10.0,
                            "cumulative_score": 38.5,
                            "global_path_score": 42.0,
                            "sequence_previous_slide_index": 1,
                            "sequence_reason": "sequence_advance_bonus",
                            "sequence_rejection_reason": "",
                            "sequence_score_gap": 0.0,
                        },
                        {
                            "source": "ppt_slide",
                            "slide_index": 9,
                            "sequence_selected": False,
                            "base_score": 10.5,
                            "sequence_score": -4.0,
                            "cumulative_score": 20.5,
                            "global_path_score": 24.0,
                            "sequence_previous_slide_index": 1,
                            "sequence_reason": "sequence_large_jump_penalty",
                            "sequence_rejection_reason": "lost_to_slide_002_on_global_path_score",
                            "sequence_score_gap": 18.0,
                        },
                    ],
                }
            ],
        },
    )

    assert "# PPT Alignment Debug: RNN基本原理" in markdown
    assert "selected_path_trace" in markdown
    assert "lost_to_slide_002_on_global_path_score" in markdown
    assert "slide `9`" in markdown


def test_build_note_outline_builds_chapter_and_sections_from_alignment() -> None:
    outline = build_note_outline(
        title="4.1BERT-2训练",
        visual_alignment={
            "segments": [
                {
                    "segment_id": "segment_001",
                    "start": 0.0,
                    "end": 60.0,
                    "summary": "本段讲解 BERT 预训练里的 CLS 与句子组织方式。",
                    "cleaned_text": "清洗稿一。",
                    "raw_text": "raw1",
                },
                {
                    "segment_id": "segment_002",
                    "start": 60.0,
                    "end": 120.0,
                    "summary": "本段说明 CLS 在分类任务中的作用。",
                    "cleaned_text": "清洗稿二。",
                    "raw_text": "raw2",
                },
            ]
        },
    )

    assert outline["chapter_count"] == 1
    assert outline["chapters"][0]["title"] == "BERT-2训练"
    assert outline["chapters"][0]["sections"][0]["title"].startswith("BERT 预训练里的 CLS")
    assert outline["chapters"][0]["sections"][0]["time_label"] == "00:00:00"


def test_build_note_outline_strips_boilerplate_and_keeps_complete_heading() -> None:
    outline = build_note_outline(
        title="4.1BERT-2训练",
        visual_alignment={
            "segments": [
                {
                    "segment_id": "segment_001",
                    "start": 0.0,
                    "end": 60.0,
                    "summary": "本段讲解了 BERT 输入构造里 CLS 与 SEP 的作用，以及句子结束标记。",
                    "cleaned_text": "这里继续说明 CLS 与 SEP 的作用。",
                    "raw_text": "raw1",
                }
            ]
        },
    )

    assert outline["chapters"][0]["sections"][0]["title"] == "BERT 输入构造里 CLS 与 SEP 的作用"


def test_build_note_blocks_organizes_summary_points_and_visuals() -> None:
    outline = {
        "chapters": [
            {
                "chapter_id": "chapter_001",
                "sections": [
                    {
                        "section_id": "section_001",
                        "segment_id": "segment_001",
                        "title": "讲解 CLS",
                        "time_label": "00:00:00",
                        "start": 0.0,
                        "end": 60.0,
                    }
                ],
            }
        ]
    }
    blocks = build_note_blocks(
        note_outline=outline,
        visual_alignment={
            "segments": [
                {
                    "segment_id": "segment_001",
                    "start": 0.0,
                    "end": 60.0,
                    "summary": "本段讲解 BERT 输入里的 CLS 标记。",
                    "cleaned_text": "每一个句子的最开始，我们加了 CLS。它表示分类。注意不要把 CLS 当作普通 token。",
                    "raw_text": "raw",
                    "issues": ["修正 Bort -> BERT"],
                    "selected_visuals": [{"relative_path": "assets/unit.jpg", "timestamp": 12.0, "ocr_text": "CLS"}],
                    "selection_mode": "visual_unit_nearby",
                }
            ]
        },
    )

    assert blocks["block_count"] == 1
    assert blocks["blocks"][0]["title"] == "讲解 CLS"
    assert blocks["blocks"][0]["summary"] == "本段讲解 BERT 输入里的 CLS 标记。"
    assert blocks["blocks"][0]["explanation_points"]
    assert blocks["blocks"][0]["key_points"][0].startswith("BERT 输入里的 CLS 标记")
    assert blocks["blocks"][0]["concepts"] == ["讲解 CLS"]
    assert blocks["blocks"][0]["pitfalls"] == []
    assert blocks["blocks"][0]["block_kind"] == "illustrated"
    assert blocks["blocks"][0]["knowledge_units"]
    assert blocks["blocks"][0]["difficulty"] in {"低", "中", "高"}
    assert blocks["blocks"][0]["timestamp_ref"] == "[00:00]"
    assert blocks["blocks"][0]["visuals"][0]["relative_path"] == "assets/unit.jpg"


def test_build_note_outline_prefers_ppt_slide_title_when_present() -> None:
    outline = build_note_outline(
        title="1.1.1RNN基本原理",
        visual_alignment={
            "segments": [
                {
                    "segment_id": "segment_001",
                    "start": 0.0,
                    "end": 60.0,
                    "summary": "这里讲循环神经网络的概念。",
                    "cleaned_text": "RNN 会引入隐藏状态。",
                    "raw_text": "raw1",
                    "selected_visuals": [
                        {
                            "source": "ppt_slide",
                            "slide_title": "循环神经网络（RNN）",
                            "relative_path": "slides_preview/slide_001.png",
                        }
                    ],
                }
            ]
        },
    )

    assert outline["chapters"][0]["sections"][0]["title"] == "循环神经网络（RNN）"
    assert outline["chapters"][0]["sections"][0]["source_title"] == "循环神经网络（RNN）"


def test_build_note_outline_ignores_generic_slide_titles() -> None:
    outline = build_note_outline(
        title="1.1.1RNN基本原理",
        visual_alignment={
            "segments": [
                {
                    "segment_id": "segment_001",
                    "start": 0.0,
                    "end": 60.0,
                    "summary": "这里讲循环神经网络的概念。",
                    "cleaned_text": "RNN 会引入隐藏状态。",
                    "raw_text": "raw1",
                    "selected_visuals": [
                        {
                            "source": "ppt_slide",
                            "slide_title": "Slide 22",
                            "relative_path": "slides_preview/slide_022.png",
                        }
                    ],
                }
            ]
        },
    )

    assert outline["chapters"][0]["sections"][0]["title"] != "Slide 22"


def test_build_note_blocks_extracts_textbook_style_knowledge_units() -> None:
    outline = {
        "chapters": [
            {
                "chapter_id": "chapter_001",
                "sections": [
                    {
                        "section_id": "section_001",
                        "segment_id": "segment_001",
                        "title": "RNN 的隐藏状态机制",
                        "time_label": "00:01:06",
                        "start": 66.0,
                        "end": 120.0,
                    }
                ],
            }
        ]
    }
    blocks = build_note_blocks(
        note_outline=outline,
        visual_alignment={
            "segments": [
                {
                    "segment_id": "segment_001",
                    "start": 66.0,
                    "end": 120.0,
                    "summary": "本段说明 RNN 通过隐藏状态保留历史信息。",
                    "cleaned_text": "RNN 在当前时刻同时接收当前输入和上一时刻的隐藏状态。它和传统 DNN 的区别在于能够保留时序信息。注意不要把隐藏状态误当作输出层。",
                    "raw_text": "raw",
                    "issues": ["修正 纪念 -> 节点"],
                    "selected_visuals": [
                        {
                            "source": "ppt_slide",
                            "slide_title": "RNN 的隐藏状态",
                            "relative_path": "slides_preview/slide_002.png",
                            "timestamp": 66.0,
                            "ocr_text": "hidden state recurrent neural network",
                        }
                    ],
                    "selection_mode": "ppt_slide_match",
                }
            ]
        },
    )

    block = blocks["blocks"][0]
    assert block["block_kind"] == "illustrated"
    assert block["mechanism_points"]
    assert block["comparison_points"] == []
    assert block["pitfalls"] == []
    assert block["knowledge_units"]
    assert block["primary_visual_source"] == "ppt_slide"
    assert block["primary_visual_title"] == "RNN 的隐藏状态"


def test_build_note_blocks_builds_reader_facing_fact_points() -> None:
    outline = {
        "chapters": [
            {
                "chapter_id": "chapter_001",
                "sections": [
                    {
                        "section_id": "section_001",
                        "segment_id": "segment_001",
                        "title": "掩码词预测",
                        "time_label": "00:04:13",
                        "start": 253.0,
                        "end": 312.0,
                    }
                ],
            }
        ]
    }
    blocks = build_note_blocks(
        note_outline=outline,
        visual_alignment={
            "segments": [
                {
                    "segment_id": "segment_001",
                    "start": 253.0,
                    "end": 312.0,
                    "summary": "本段讲掩码词预测的输出链路。",
                    "cleaned_text": "随机遮盖部分 token，再输入 Transformer 编码器。BERT-base 使用 12 层编码器。最后经过线性层和 softmax 输出概率分布。",
                    "raw_text": "raw",
                    "issues": [],
                    "selected_visuals": [],
                    "selection_mode": "frame_fallback",
                }
            ]
        },
    )

    fact_points = blocks["blocks"][0]["fact_points"]
    assert any("随机遮盖部分 token" in point for point in fact_points)
    assert any("BERT-base 使用 12 层编码器" in point for point in fact_points)
    assert any("softmax 输出概率分布" in point for point in fact_points)


def test_build_note_blocks_extracts_formula_candidates_from_text_and_ocr() -> None:
    outline = {
        "chapters": [
            {
                "chapter_id": "chapter_001",
                "sections": [
                    {
                        "section_id": "section_001",
                        "segment_id": "segment_001",
                        "title": "softmax 概率输出",
                        "time_label": "00:04:19",
                        "start": 259.0,
                        "end": 312.0,
                    }
                ],
            }
        ]
    }
    blocks = build_note_blocks(
        note_outline=outline,
        visual_alignment={
            "segments": [
                {
                    "segment_id": "segment_001",
                    "start": 259.0,
                    "end": 312.0,
                    "summary": "本段讲 softmax 输出概率分布。",
                    "cleaned_text": "最后通过线性层得到 z_i，再计算 P(y_i)=e^{z_i}/sum_j e^{z_j}。",
                    "raw_text": "raw",
                    "issues": [],
                    "selected_visuals": [
                        {
                            "relative_path": "assets/unit.jpg",
                            "timestamp": 278.0,
                            "ocr_text": "P(y_i)=e^{z_i}/sum_j e^{z_j}",
                        }
                    ],
                    "selection_mode": "visual_unit_nearby",
                }
            ]
        },
    )

    assert blocks["blocks"][0]["formula_candidates"]
    assert blocks["blocks"][0]["formula_candidates"][0]["latex"] == "P(y_i)=e^{z_i}/sum_j e^{z_j}"
    assert "P(y_i)=e^{z_i}/sum_j e^{z_j}" in blocks["blocks"][0]["formula_candidates"][0]["evidence"]


def test_build_note_generation_prompt_requires_timestamps_definition_cards_tables_and_latex() -> None:
    prompt = build_note_generation_prompt(
        title="4.1BERT-2训练",
        source_video="4.1BERT-2训练.mp4",
        note_outline={
            "chapters": [
                {
                    "index_label": "一",
                    "title": "BERT-2训练",
                    "time_label": "00:00:00",
                    "sections": [{"index": 1, "title": "CLS 的作用", "time_label": "00:00:58"}],
                }
            ]
        },
        note_blocks={
            "blocks": [
                {
                    "title": "CLS 的作用",
                    "timestamp_ref": "[00:58]",
                    "summary": "CLS 用于分类。",
                    "key_points": ["CLS 聚合整句信息。"],
                    "definitions": ["CLS 是句级分类标记。"],
                    "pitfalls": ["不要把 CLS 当作普通 token。"],
                    "difficulty": "中",
                    "formula_candidates": [
                        {
                            "name": "softmax",
                            "latex": r"P(y_i)=\frac{e^{z_i}}{\sum_j e^{z_j}}",
                            "evidence": "softmax probability distribution",
                        }
                    ],
                }
            ]
        },
        corrections=[{"raw": "Bort", "cleaned": "BERT", "reason": "term"}],
    )

    assert "必须输出多级列表" in prompt
    assert "提纲式学习讲义" in prompt
    assert "整体结构要尽量接近教材提纲" in prompt
    assert "核心定义卡片" in prompt
    assert "知识点总结表格" in prompt
    assert "[02:48]" in prompt
    assert "必须参考 OCR 证据写出正确的 LaTeX 公式" in prompt
    assert "默认每个小节先写 3-5 个高信息密度 bullet" in prompt
    assert "不要为每一节强行凑满所有栏目" in prompt
    assert "同一公式不要在多个相邻小节反复整块重复" in prompt
    assert "全文必须出现 `## 知识小结`、`核心定义卡片` 和 `知识框架` 的原文字眼" in prompt
    assert "输入里的标题只是素材线索，不是最终标题定稿" in prompt
    assert "多数小节可直接写成 `1）2）3）` 的提纲子点" in prompt
    assert "不要输出“术语修正”" in prompt
    assert "不要解释为什么没写" in prompt
    assert r"P(y_i)=\frac{e^{z_i}}{\sum_j e^{z_j}}" in prompt


def test_build_note_generation_prompt_filters_internal_correction_language_from_reader_facing_summary() -> None:
    prompt = build_note_generation_prompt(
        title="RNN",
        source_video="rnn.mp4",
        note_outline={
            "chapters": [
                {
                    "index_label": "一",
                    "title": "RNN基本原理",
                    "time_label": "00:00:00",
                    "sections": [{"index": 1, "title": "RNN 的记忆机制", "start": 66.0}],
                }
            ]
        },
        note_blocks={
            "blocks": [
                {
                    "title": "RNN 的记忆机制",
                    "timestamp_ref": "[01:06]",
                    "summary": "RNN 当前时刻依赖当前输入与前一时刻状态。",
                    "key_points": [
                        "RNN 当前时刻依赖当前输入与前一时刻状态。",
                        "修正了“并停”为“并行”等明显口误，理顺句子断句。",
                    ],
                    "explanation_points": [
                        "模型会把前面的信息带到当前计算。",
                        "OCR 证据不足, 不补写公式。",
                    ],
                    "formula_candidates": [],
                    "visuals": [],
                }
            ]
        },
        corrections=[{"raw": "并停", "cleaned": "并行", "reason": "口误"}],
    )

    assert "修正了“并停”为“并行”等明显口误" not in prompt
    assert "OCR 证据不足, 不补写公式" not in prompt
    assert "纠错背景" not in prompt
    assert "RNN 当前时刻依赖当前输入与前一时刻状态。" in prompt
    assert "模型会把前面的信息带到当前计算。" in prompt
    assert "讲解素材：" not in prompt


def test_build_note_generation_prompt_keeps_reader_facing_points_without_sentence_rewrite() -> None:
    prompt = build_note_generation_prompt(
        title="RNN",
        source_video="rnn.mp4",
        note_outline={
            "chapters": [
                {
                    "index_label": "一",
                    "title": "RNN基本原理",
                    "time_label": "00:00:00",
                    "sections": [{"index": 1, "title": "RNN 的记忆机制", "start": 0.0}],
                }
            ]
        },
        note_blocks={
            "blocks": [
                {
                    "title": "RNN 的记忆机制",
                    "timestamp_ref": "[00:00]",
                    "summary": "RNN 会把前文信息带到当前计算。",
                    "key_points": [
                        "好，这次我们和大家一起来研究一下这个 RNN，也就是循环神经网络。",
                        "在正式介绍循环神经网络之前，我们先看一下普通的 DNN 神经网络它的运行机制。",
                    ],
                    "explanation_points": [
                        "我们可以看到，这是一个传统的 DNN 网络架构图。",
                        "也就是说，模型会把前面的信息带到当前计算。",
                    ],
                    "formula_candidates": [],
                    "visuals": [],
                }
            ]
        },
        corrections=[],
    )

    assert "好，这次我们和大家一起来研究一下这个 RNN，也就是循环神经网络。" in prompt
    assert "在正式介绍循环神经网络之前，我们先看一下普通的 DNN 神经网络它的运行机制。" in prompt
    assert "我们可以看到，这是一个传统的 DNN 网络架构图。" in prompt
    assert "模型会把前面的信息带到当前计算。" in prompt


def test_build_note_generation_prompt_preserves_material_titles_and_delegates_rewrite_to_codex() -> None:
    prompt = build_note_generation_prompt(
        title="4.1BERT-2训练",
        source_video="4.1BERT-2训练.mp4",
        note_outline={
            "chapters": [
                {
                    "index_label": "一",
                    "title": "BERT-2训练",
                    "time_label": "00:00:00",
                    "sections": [
                        {"index": 1, "title": "Bert PART", "start": 0.0},
                        {"index": 2, "title": "默认取第 0 个位置的 CLS 表示连接分类层和 softmax", "start": 390.0},
                    ],
                }
            ]
        },
        note_blocks={
            "blocks": [
                {
                    "title": "Bert PART",
                    "timestamp_ref": "[00:00]",
                    "summary": "本段讲解了 BERT 在预训练时如何处理句子对，并说明了句首 CLS 标记的作用。",
                    "key_points": ["句对输入会加入 CLS 和 SEP。"],
                    "definitions": ["BERT 在预训练时会组织句对输入。"],
                    "formula_candidates": [],
                },
                {
                    "title": "默认取第 0 个位置的 CLS 表示连接分类层和 softmax",
                    "timestamp_ref": "[06:30]",
                    "summary": "本段说明默认取第 0 个位置的 CLS 表示连接分类层和 softmax。",
                    "key_points": ["默认使用 index 为 0 的 CLS 表示做分类。"],
                    "definitions": ["index 为 0 的 CLS 常作为分类读取位。"],
                    "formula_candidates": [],
                },
            ]
        },
        corrections=[],
    )

    assert "- [00:00] BERT PART" in prompt
    assert "BERT" in prompt
    assert "- [06:30] 默认取第 0 个位置的 CLS 表示连接分类层和 softmax" in prompt
    assert "第 0 个位置" in prompt
    assert "标题只是素材线索" in prompt


def test_build_note_generation_prompt_rewrites_visual_paths_relative_to_note_file() -> None:
    prompt = build_note_generation_prompt(
        title="4.1BERT-2训练",
        source_video="4.1BERT-2训练.mp4",
        note_outline={
            "chapters": [
                {
                    "index_label": "一",
                    "title": "BERT-2训练",
                    "time_label": "00:00:00",
                    "sections": [{"index": 1, "title": "CLS 的作用", "start": 58.0}],
                }
            ]
        },
        note_blocks={
            "blocks": [
                {
                    "title": "CLS 的作用",
                    "timestamp_ref": "[00:58]",
                    "summary": "CLS 用于分类。",
                    "key_points": ["CLS 聚合整句信息。"],
                    "definitions": ["CLS 是句级分类标记。"],
                    "formula_candidates": [],
                    "visuals": [
                        {
                            "relative_path": "pipeline/01_media/visual/visual_candidates/frame_001.jpg",
                            "timestamp": 58.0,
                        }
                    ],
                }
            ]
        },
        corrections=[],
    )

    assert "pipeline/01_media/visual/visual_candidates/frame_001.jpg" not in prompt
    assert "../01_media/visual/visual_candidates/frame_001.jpg" in prompt


def test_build_note_markdown_includes_knowledge_summary_table() -> None:
    markdown = build_note_markdown(
        title="4.1BERT-2训练",
        source_video="4.1BERT-2训练.mp4",
        duration_seconds=120.0,
        transcript_excerpt=[],
        cleaned_segments=["CLS 用于分类任务。"],
        review_segments=[
            {
                "segment_id": "segment_001",
                "start": 0.0,
                "end": 60.0,
                "label": "00:00:00-00:01:00",
                "text": "原始片段",
            }
        ],
        segment_reviews=[
            {
                "segment_id": "segment_001",
                "start": 0.0,
                "end": 60.0,
                "summary": "本段讲解 CLS 在分类任务中的作用。",
                "issues": ["修正 Bort -> BERT"],
                "status": "done",
            }
        ],
        corrections=[],
        frames=[],
    )

    assert "## 知识小结" in markdown
    assert "| 小节 | 核心内容 | 易错点 | 难度 |" in markdown
    assert "| CLS 在分类任务中的作用 |" in markdown
    assert "修正 Bort -> BERT" not in markdown


def test_build_note_markdown_rewrites_visual_paths_relative_to_note_file() -> None:
    markdown = build_note_markdown(
        title="4.1BERT-2训练",
        source_video="4.1BERT-2训练.mp4",
        duration_seconds=120.0,
        transcript_excerpt=[],
        cleaned_segments=[],
        review_segments=[],
        segment_reviews=[],
        corrections=[],
        note_outline={
            "chapters": [
                {
                    "chapter_id": "chapter_001",
                    "index_label": "一",
                    "title": "BERT-2训练",
                    "time_label": "00:00:00",
                    "sections": [
                        {
                            "section_id": "section_001",
                            "index": 1,
                            "title": "CLS 的作用",
                            "start": 58.0,
                        }
                    ],
                }
            ]
        },
        note_blocks={
            "blocks": [
                {
                    "section_id": "section_001",
                    "timestamp_ref": "[00:58]",
                    "title": "CLS 的作用",
                    "summary": "CLS 用于分类。",
                    "concepts": ["CLS 的作用"],
                    "definitions": ["CLS 是句级分类标记。"],
                    "fact_points": ["CLS 聚合整句信息。"],
                    "formula_candidates": [],
                    "pitfalls": [],
                    "explanation_points": [],
                    "visuals": [
                        {
                            "relative_path": "pipeline/01_media/visual/visual_candidates/frame_001.jpg",
                            "timestamp": 58.0,
                        }
                    ],
                }
            ]
        },
        frames=[
            {
                "relative_path": "pipeline/01_media/visual/visual_candidates/frame_001.jpg",
                "timestamp": 58.0,
            }
        ],
    )

    assert "![00:00:58](pipeline/01_media/visual/visual_candidates/frame_001.jpg)" not in markdown
    assert "![00:00:58](../01_media/visual/visual_candidates/frame_001.jpg)" in markdown


def test_preprocess_image_for_ocr_trims_border_and_enhances_image(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    output = tmp_path / "preprocessed.png"
    image = Image.new("RGB", (120, 120), "white")
    for x in range(30, 90):
        for y in range(35, 85):
            image.putpixel((x, y), (0, 0, 0))
    image.save(source)

    metadata = preprocess_image_for_ocr(source, output)

    assert output.exists()
    processed = Image.open(output)
    assert processed.mode == "L"
    assert processed.size[0] < 120
    assert processed.size[1] < 120
    assert "autocontrast" in metadata["steps"]
    assert metadata["cropped"] is True


def test_annotate_scene_change_scores_marks_large_visual_jump(tmp_path: Path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    third = tmp_path / "third.png"
    Image.new("RGB", (60, 60), "white").save(first)
    Image.new("RGB", (60, 60), "white").save(second)
    Image.new("RGB", (60, 60), "black").save(third)

    frames = annotate_scene_change_scores(
        [
            {"timestamp": 0.0, "path": str(first), "relative_path": "a.png"},
            {"timestamp": 10.0, "path": str(second), "relative_path": "b.png"},
            {"timestamp": 20.0, "path": str(third), "relative_path": "c.png"},
        ]
    )

    assert frames[0]["scene_change_score"] >= 0.0
    assert frames[1]["scene_change_score"] < frames[2]["scene_change_score"]


def test_ocr_frames_records_preprocessed_source_and_quality(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "frame.png"
    Image.new("RGB", (80, 80), "white").save(image_path)

    monkeypatch.setattr(video_to_notes, "run_tesseract", lambda path: "BERT CLS embedding")

    results = video_to_notes.ocr_frames(
        [{"timestamp": 12.0, "path": str(image_path), "relative_path": "assets/frame_001.jpg"}],
        tmp_path / "ocr.json",
    )

    assert results[0]["ocr_text"] == "BERT CLS embedding"
    assert results[0]["ocr_source"] == "preprocessed"
    assert results[0]["ocr_quality_score"] > 0


def test_extract_frames_at_timestamps_avoids_filename_collisions_across_multiple_calls(monkeypatch, tmp_path: Path) -> None:
    assets_dir = tmp_path / "visual_candidates"

    def fake_run_command(command):
        Path(command[-1]).write_bytes(b"jpg")

    monkeypatch.setattr(video_to_notes, "run_command", fake_run_command)

    first = video_to_notes.extract_frames_at_timestamps(
        Path("video.mp4"),
        assets_dir,
        timestamps=[10, 20],
    )
    second = video_to_notes.extract_frames_at_timestamps(
        Path("video.mp4"),
        assets_dir,
        timestamps=[30],
    )

    assert first[0]["relative_path"] == "visual_candidates/frame_001.jpg"
    assert first[1]["relative_path"] == "visual_candidates/frame_002.jpg"
    assert second[0]["relative_path"] == "visual_candidates/frame_003.jpg"


def test_build_visual_alignment_records_selected_visuals_and_selection_mode() -> None:
    payload = build_visual_alignment(
        cleaned_segments=["这里在讲 CLS 和 BERT 的输入表示。"],
        review_segments=[
            {
                "segment_id": "segment_001",
                "start": 0.0,
                "end": 60.0,
                "label": "00:00:00-00:01:00",
                "text": "原始片段",
                "ocr_hints": ["BERT CLS"],
            }
        ],
        segment_reviews=[
            {
                "segment_id": "segment_001",
                "start": 0.0,
                "end": 60.0,
                "summary": "本段讲 BERT 输入里的 CLS 标记。",
                "issues": ["修正 Bort -> BERT"],
                "status": "done",
            }
        ],
        visual_units=[
            {
                "unit_id": "visual_unit_001",
                "start": 10.0,
                "end": 20.0,
                "representative_frame": "assets/unit_frame_001.jpg",
                "representative_timestamp": 12.0,
                "ocr_text": "BERT CLS embedding",
                "ocr_len": 18,
                "is_low_value": False,
            }
        ],
        frames=[
            {"timestamp": 12.0, "relative_path": "assets/frame_001.jpg", "ocr_text": "irrelevant raw frame"},
        ],
    )

    assert payload["segment_count"] == 1
    assert payload["segments_with_visuals"] == 1
    assert payload["segments"][0]["selection_mode"] == "visual_unit_nearby"
    assert payload["segments"][0]["borrow_reason"] == ""
    assert payload["segments"][0]["reject_reason"] == ""
    assert payload["segments"][0]["selected_visuals"][0]["source"] == "visual_unit"
    assert payload["segments"][0]["selected_visuals"][0]["unit_id"] == "visual_unit_001"
    assert payload["segments"][0]["selected_visuals"][0]["relative_path"] == "assets/unit_frame_001.jpg"
    assert payload["segments"][0]["selected_visuals"][0]["text_score"] > 0
    assert payload["segments"][0]["selected_visuals"][0]["time_score"] < 0
    assert payload["segments"][0]["selected_visuals"][0]["quality_score"] == 0.0


def test_build_visual_alignment_uses_frame_fallback_when_no_visual_unit_matches() -> None:
    payload = build_visual_alignment(
        cleaned_segments=["这里在讲 CLS 和 BERT 的输入表示。"],
        review_segments=[
            {
                "segment_id": "segment_001",
                "start": 0.0,
                "end": 60.0,
                "label": "00:00:00-00:01:00",
                "text": "原始片段",
                "ocr_hints": ["BERT CLS"],
            }
        ],
        segment_reviews=[
            {
                "segment_id": "segment_001",
                "start": 0.0,
                "end": 60.0,
                "summary": "本段讲 BERT 输入里的 CLS 标记。",
                "issues": [],
                "status": "done",
            }
        ],
        visual_units=[],
        frames=[
            {"timestamp": 12.0, "relative_path": "assets/frame_001.jpg", "ocr_text": "BERT CLS embedding"},
        ],
    )

    assert payload["segments"][0]["selection_mode"] == "frame_fallback"
    assert payload["segments"][0]["borrow_reason"] == "no visual-unit match, fell back to raw frames"
    assert payload["segments"][0]["reject_reason"] == ""
    assert payload["segments"][0]["selected_visuals"][0]["source"] == "frame"
    assert payload["segments"][0]["selected_visuals"][0]["relative_path"] == "assets/frame_001.jpg"
    assert payload["segments"][0]["selected_visuals"][0]["time_score"] > 0


def test_build_visual_alignment_suppresses_all_low_value_visual_units() -> None:
    payload = build_visual_alignment(
        cleaned_segments=["这里在讲 CLS 和 BERT 的输入表示。"],
        review_segments=[
            {
                "segment_id": "segment_001",
                "start": 60.0,
                "end": 120.0,
                "label": "00:01:00-00:02:00",
                "text": "原始片段",
                "ocr_hints": ["BERT CLS"],
            }
        ],
        segment_reviews=[
            {
                "segment_id": "segment_001",
                "start": 60.0,
                "end": 120.0,
                "summary": "本段讲 BERT 输入里的 CLS 标记。",
                "issues": [],
                "status": "done",
            }
        ],
        visual_units=[
            {
                "unit_id": "visual_unit_001",
                "start": 80.0,
                "end": 82.0,
                "representative_frame": "assets/weak_001.jpg",
                "representative_timestamp": 81.0,
                "ocr_text": "BERT WA CLS",
                "ocr_len": 11,
                "is_low_value": True,
            },
            {
                "unit_id": "visual_unit_002",
                "start": 95.0,
                "end": 97.0,
                "representative_frame": "assets/weak_002.jpg",
                "representative_timestamp": 96.0,
                "ocr_text": "BERT WE CLS",
                "ocr_len": 11,
                "is_low_value": True,
            },
        ],
        frames=[],
    )

    assert payload["segments"][0]["selection_mode"] == "suppressed_low_value"
    assert payload["segments"][0]["borrow_reason"] == ""
    assert (
        payload["segments"][0]["reject_reason"]
        == "nearby matched visual units were all low-value and no nearby high-quality visual met semantic overlap"
    )
    assert payload["segments"][0]["selected_visuals"] == []
    assert payload["segments"][0]["candidate_visuals"]
    assert payload["segments"][0]["has_useful_visual"] is False


def test_build_visual_alignment_borrows_nearby_high_quality_visual_when_nearby_matches_are_low_value() -> None:
    payload = build_visual_alignment(
        cleaned_segments=["这里在讲 CLS 和 BERT 的输入表示。"],
        review_segments=[
            {
                "segment_id": "segment_001",
                "start": 60.0,
                "end": 120.0,
                "label": "00:01:00-00:02:00",
                "text": "原始片段",
                "ocr_hints": ["BERT CLS"],
            }
        ],
        segment_reviews=[
            {
                "segment_id": "segment_001",
                "start": 60.0,
                "end": 120.0,
                "summary": "本段讲 BERT 输入里的 CLS 标记。",
                "issues": [],
                "status": "done",
            }
        ],
        visual_units=[
            {
                "unit_id": "visual_unit_001",
                "start": 80.0,
                "end": 82.0,
                "representative_frame": "assets/weak_001.jpg",
                "representative_timestamp": 81.0,
                "ocr_text": "BERT WA CLS",
                "ocr_len": 11,
                "is_low_value": True,
            },
            {
                "unit_id": "visual_unit_002",
                "start": 145.0,
                "end": 148.0,
                "representative_frame": "assets/strong.jpg",
                "representative_timestamp": 146.0,
                "ocr_text": "CLS embedding token_type position",
                "ocr_len": 29,
                "is_low_value": False,
            },
        ],
        frames=[],
    )

    assert payload["segments"][0]["selection_mode"] == "borrowed_nearby_visual"
    assert payload["segments"][0]["has_useful_visual"] is True
    assert (
        payload["segments"][0]["borrow_reason"]
        == "borrowed nearby non-low-value visual units that also met minimum semantic overlap"
    )
    assert payload["segments"][0]["reject_reason"] == ""
    assert payload["segments"][0]["selected_visuals"][0]["relative_path"] == "assets/strong.jpg"
    assert "nearby_high_quality_borrow" in payload["segments"][0]["selected_visuals"][0]["selection_reason"]
    assert payload["segments"][0]["selected_visuals"][0]["text_score"] > 0.0
    assert payload["segments"][0]["selected_visuals"][0]["time_score"] > 0
    assert payload["segments"][0]["selected_visuals"][0]["quality_score"] == 5.0
    assert "text_overlap" in payload["segments"][0]["selected_visuals"][0]["selection_reason"]


def test_build_visual_alignment_does_not_borrow_high_quality_visual_without_semantic_overlap() -> None:
    payload = build_visual_alignment(
        cleaned_segments=["这里在讲 CLS 和 BERT 的输入表示。"],
        review_segments=[
            {
                "segment_id": "segment_001",
                "start": 60.0,
                "end": 120.0,
                "label": "00:01:00-00:02:00",
                "text": "原始片段",
                "ocr_hints": ["BERT CLS"],
            }
        ],
        segment_reviews=[
            {
                "segment_id": "segment_001",
                "start": 60.0,
                "end": 120.0,
                "summary": "本段讲 BERT 输入里的 CLS 标记。",
                "issues": [],
                "status": "done",
            }
        ],
        visual_units=[
            {
                "unit_id": "visual_unit_001",
                "start": 80.0,
                "end": 82.0,
                "representative_frame": "assets/weak_001.jpg",
                "representative_timestamp": 81.0,
                "ocr_text": "BERT WA CLS",
                "ocr_len": 11,
                "is_low_value": True,
            },
            {
                "unit_id": "visual_unit_002",
                "start": 145.0,
                "end": 148.0,
                "representative_frame": "assets/strong.jpg",
                "representative_timestamp": 146.0,
                "ocr_text": "embedding token_type position",
                "ocr_len": 29,
                "is_low_value": False,
            },
        ],
        frames=[],
    )

    assert payload["segments"][0]["selection_mode"] in {"suppressed_low_value", "none"}
    assert payload["segments"][0]["selected_visuals"] == []


def test_build_visual_alignment_does_not_borrow_on_generic_bert_token_alone() -> None:
    payload = build_visual_alignment(
        cleaned_segments=["这里在讲 CLS 和分类任务。"],
        review_segments=[
            {
                "segment_id": "segment_001",
                "start": 60.0,
                "end": 120.0,
                "label": "00:01:00-00:02:00",
                "text": "原始片段",
                "ocr_hints": ["CLS 分类"],
            }
        ],
        segment_reviews=[
            {
                "segment_id": "segment_001",
                "start": 60.0,
                "end": 120.0,
                "summary": "本段讲 CLS 的分类作用。",
                "issues": [],
                "status": "done",
            }
        ],
        visual_units=[
            {
                "unit_id": "visual_unit_001",
                "start": 80.0,
                "end": 82.0,
                "representative_frame": "assets/weak_001.jpg",
                "representative_timestamp": 81.0,
                "ocr_text": "CLS WA",
                "ocr_len": 6,
                "is_low_value": True,
            },
            {
                "unit_id": "visual_unit_002",
                "start": 90.0,
                "end": 92.0,
                "representative_frame": "assets/generic.jpg",
                "representative_timestamp": 91.0,
                "ocr_text": "BERT transformer",
                "ocr_len": 16,
                "is_low_value": False,
            },
        ],
        frames=[],
    )

    assert payload["segments"][0]["selection_mode"] in {"suppressed_low_value", "none"}
    assert payload["segments"][0]["selected_visuals"] == []


def test_build_visual_alignment_rejects_weak_global_visual_unit_matches() -> None:
    payload = build_visual_alignment(
        cleaned_segments=["这里在讲 CLS 和分类任务。"],
        review_segments=[
            {
                "segment_id": "segment_001",
                "start": 0.0,
                "end": 30.0,
                "label": "00:00:00-00:00:30",
                "text": "原始片段",
                "ocr_hints": ["CLS 分类"],
            }
        ],
        segment_reviews=[
            {
                "segment_id": "segment_001",
                "start": 0.0,
                "end": 30.0,
                "summary": "本段讲 CLS 的分类作用。",
                "issues": [],
                "status": "done",
            }
        ],
        visual_units=[
            {
                "unit_id": "visual_unit_001",
                "start": 180.0,
                "end": 182.0,
                "representative_frame": "assets/far.jpg",
                "representative_timestamp": 181.0,
                "ocr_text": "embedding CLS",
                "ocr_len": 13,
                "is_low_value": False,
            }
        ],
        frames=[],
    )

    assert payload["segments"][0]["selection_mode"] == "none"
    assert payload["segments"][0]["selected_visuals"] == []


def test_build_note_markdown_omits_images_when_alignment_suppresses_low_value_visuals() -> None:
    markdown = build_note_markdown(
        title="BERT",
        source_video="bert.mp4",
        duration_seconds=120.0,
        transcript_excerpt=[],
        cleaned_segments=["这里在讲 CLS 和 BERT 的输入表示。"],
        review_segments=[
            {
                "segment_id": "segment_001",
                "start": 60.0,
                "end": 120.0,
                "label": "00:01:00-00:02:00",
                "text": "原始片段",
                "ocr_hints": ["BERT CLS"],
            }
        ],
        segment_reviews=[
            {
                "segment_id": "segment_001",
                "start": 60.0,
                "end": 120.0,
                "summary": "本段讲 BERT 输入里的 CLS 标记。",
                "issues": [],
                "status": "done",
            }
        ],
        corrections=[],
        visual_alignment={
            "segment_count": 1,
            "segments_with_visuals": 0,
            "segments": [
                {
                    "segment_id": "segment_001",
                    "label": "00:01:00-00:02:00",
                    "start": 60.0,
                    "end": 120.0,
                    "summary": "本段讲 BERT 输入里的 CLS 标记。",
                    "issues": [],
                    "cleaned_text": "这里在讲 CLS 和 BERT 的输入表示。",
                    "raw_text": "原始片段",
                    "selection_mode": "suppressed_low_value",
                    "has_useful_visual": False,
                    "borrow_reason": "",
                    "reject_reason": "nearby matched visual units were all low-value and no nearby high-quality visual met semantic overlap",
                    "selected_visuals": [],
                    "candidate_visuals": [
                        {
                            "source": "visual_unit",
                            "unit_id": "visual_unit_001",
                            "relative_path": "assets/weak.jpg",
                            "ocr_text": "WA",
                            "is_low_value": True,
                            "score": 1.0,
                            "text_score": 10.0,
                            "time_score": -1.0,
                            "quality_score": -8.0,
                            "timestamp": 81.0,
                            "time_distance_seconds": 9.0,
                            "selection_reason": ["nearby_window", "text_overlap", "visual_unit_match"],
                        }
                    ],
                }
            ],
        },
        frames=[],
    )

    structured_notes = markdown.split("## 关键截图索引", 1)[0]
    assert "assets/weak.jpg" not in structured_notes
    assert "![00:01:21]" not in structured_notes


def test_parse_args_defaults_to_small_whisper_model() -> None:
    args = parse_args(["--input", "materials/videos/4.1BERT-2训练.mp4"])
    assert args.whisper_model == "small"
    assert args.codex_timeout_seconds == 300
    assert args.codex_review_parallelism == 5
    assert args.visual_source_mode == "auto"
    assert args.skip_codex_review is False
    assert args.skip_codex_note is False


def test_parse_args_accepts_skip_codex_note() -> None:
    args = parse_args(["--input", "materials/videos/4.1BERT-2训练.mp4", "--skip-codex-note"])
    assert args.skip_codex_note is True


def test_select_representative_frame_prefers_high_information_ocr() -> None:
    representative = select_representative_frame(
        [
            {"timestamp": 12.0, "relative_path": "assets/frame_001.jpg", "ocr_text": "BERT"},
            {"timestamp": 16.0, "relative_path": "assets/frame_002.jpg", "ocr_text": "BERT CLS embedding token_type position"},
            {"timestamp": 20.0, "relative_path": "assets/frame_003.jpg", "ocr_text": ""},
        ]
    )

    assert representative["relative_path"] == "assets/frame_002.jpg"


def test_build_visual_units_clusters_near_duplicate_frames_and_keeps_representative() -> None:
    payload = build_visual_units(
        [
            {
                "timestamp": 10.0,
                "relative_path": "assets/frame_001.jpg",
                "ocr_text": "BERT CLS embedding",
                "phash": "aaaaaaaaaaaaaaaa",
            },
            {
                "timestamp": 14.0,
                "relative_path": "assets/frame_002.jpg",
                "ocr_text": "BERT CLS embedding token_type",
                "phash": "aaaaaaaaaaaaaaab",
            },
            {
                "timestamp": 70.0,
                "relative_path": "assets/frame_003.jpg",
                "ocr_text": "optimizer loss curve",
                "phash": "bbbbbbbbbbbbbbbb",
            },
        ]
    )

    assert payload["unit_count"] == 2
    assert payload["units"][0]["frame_count"] == 2
    assert payload["units"][0]["representative_frame"] == "assets/frame_002.jpg"
    assert payload["units"][1]["representative_frame"] == "assets/frame_003.jpg"


def test_parse_slides_pdf_extracts_text_and_renders_pages(tmp_path: Path) -> None:
    pdf_path = tmp_path / "slides.pdf"
    preview_dir = tmp_path / "slides_preview"
    import fitz

    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "RNN Overview")
    document.save(pdf_path)
    document.close()

    payload = parse_slides_pdf(pdf_path, preview_dir)

    assert payload["slide_count"] == 1
    assert payload["slides"][0]["title"] == "RNN Overview"
    assert payload["slides"][0]["relative_path"] == "slides_preview/rendered/slide_001.png"
    assert (preview_dir / "rendered" / "slide_001.png").exists()


def test_sanitize_slides_payload_removes_mojibake_from_title_and_text() -> None:
    payload = {
        "slide_count": 1,
        "slides": [
            {
                "slide_id": "slide_007",
                "slide_index": 7,
                "title": "RNN整体网络结构 ǜ֩−1 ǜ֩ ǜ֩+1",
                "text": "RNN整体网络结构 ǜ֩−1 ǜ֩ ǜ֩+1 我 喜欢 打 篮球 ǜ֩+2",
                "relative_path": "slides_preview/rendered/slide_007.png",
                "image_area": 1,
                "image_frequency": 1,
                "is_low_value": False,
            }
        ],
    }

    cleaned = sanitize_slides_payload(payload)

    assert slides_payload_has_noise(payload) is True
    assert slides_payload_has_noise(cleaned) is False
    assert cleaned["slides"][0]["title"] == "RNN整体网络结构"
    assert "ǜ֩" not in cleaned["slides"][0]["text"]


def test_assess_slides_payload_for_video_requires_explicit_slides_path() -> None:
    payload = assess_slides_payload_for_video(
        {
            "slide_count": 2,
            "slides": [
                {"title": "Transformer 原理", "text": "transformer attention encoder decoder"},
                {"title": "BERT", "text": "bert pretraining mask language model"},
            ],
        },
        input_video=Path("/tmp/9.9.9图数据库基础.mp4"),
        explicit_path=False,
    )

    assert payload["usable"] is False
    assert payload["rejected_reason"] == "explicit_slides_required"
    assert payload["matched_tokens"] == []
    assert payload["match_score"] == 0


def test_assess_slides_payload_for_video_accepts_explicit_slides_path() -> None:
    payload = assess_slides_payload_for_video(
        {
            "slide_count": 2,
            "slides": [
                {"title": "RNN", "text": "rnn hidden state recurrent neural network"},
                {"title": "LSTM", "text": "lstm forget gate"},
            ],
        },
        input_video=Path("/tmp/1.1.1RNN基本原理.mp4"),
        explicit_path=True,
    )

    assert payload["usable"] is True
    assert payload["matched_tokens"] == ["explicit_path"]


def test_determine_visual_source_mode_prefers_slides_when_usable() -> None:
    assert (
        determine_visual_source_mode(
            requested_mode="auto",
            slides_payload={"usable": True},
        )
        == "slides-first"
    )


def test_determine_visual_source_mode_falls_back_to_video_when_slides_unusable() -> None:
    assert (
        determine_visual_source_mode(
            requested_mode="slides-first",
            slides_payload={"usable": False},
        )
        == "video-first"
    )


def test_resolve_slides_path_prefers_explicit_existing_path(tmp_path: Path) -> None:
    pdf = tmp_path / "course.pdf"
    pdf.write_text("pdf", encoding="utf-8")
    resolved = resolve_slides_path(Path("/tmp/video.mp4"), str(pdf))
    assert resolved == pdf.resolve()


def test_resolve_slides_path_without_explicit_path_returns_none(tmp_path: Path, monkeypatch) -> None:
    docs_dir = tmp_path / "materials" / "docs"
    docs_dir.mkdir(parents=True)
    (docs_dir / "RNN 课件.pdf").write_text("pdf", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    resolved = resolve_slides_path(Path("/tmp/1.1.1RNN基本原理.mp4"))

    assert resolved is None


def test_parse_args_accepts_codex_review_parallelism_override() -> None:
    args = parse_args(
        [
            "--input",
            "materials/videos/4.1BERT-2训练.mp4",
            "--codex-review-parallelism",
            "3",
        ]
    )
    assert args.codex_review_parallelism == 3


def test_parse_args_accepts_visual_source_mode_override() -> None:
    args = parse_args(
        [
            "--input",
            "materials/videos/4.1BERT-2训练.mp4",
            "--visual-source-mode",
            "video-first",
        ]
    )
    assert args.visual_source_mode == "video-first"


def test_build_slides_cleanup_prompt_mentions_raw_and_cleaned_indexes(tmp_path: Path) -> None:
    output_paths = plan_output_paths(Path("/tmp/4.1BERT-2训练.mp4"), tmp_path)
    prompt = build_slides_cleanup_prompt(output_paths)
    assert "slides_index.raw.json" in prompt
    assert "slides_index.json" in prompt
    assert "slides_preview/rendered/" in prompt


def test_parse_args_accepts_force_stage_flags() -> None:
    args = parse_args(
        [
            "--input",
            "materials/videos/4.1BERT-2训练.mp4",
            "--force-audio",
            "--force-transcribe",
            "--force-frames",
            "--force-ocr",
            "--force-review-artifacts",
            "--force-codex-review",
            "--force-note",
        ]
    )

    assert args.force_audio is True
    assert args.force_transcribe is True
    assert args.force_frames is True
    assert args.force_ocr is True
    assert args.force_review_artifacts is True
    assert args.force_codex_review is True
    assert args.force_note is True


def test_audio_artifact_ready_requires_non_empty_audio(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    assert audio_artifact_ready(audio) is False
    audio.write_text("", encoding="utf-8")
    assert audio_artifact_ready(audio) is False
    audio.write_bytes(b"wav")
    assert audio_artifact_ready(audio) is True


def test_transcript_artifacts_ready_requires_all_transcript_outputs(tmp_path: Path) -> None:
    output_paths = plan_output_paths(Path("/tmp/4.1BERT-2训练.mp4"), tmp_path)
    video_to_notes.ensure_dirs(output_paths)
    Path(output_paths["transcript_txt"]).write_text("hello", encoding="utf-8")
    Path(output_paths["transcript_json"]).write_text(json.dumps({"segments": [{"start": 0.0, "end": 1.0, "text": "hello"}]}), encoding="utf-8")
    Path(output_paths["transcript_srt"]).write_text("1", encoding="utf-8")
    Path(output_paths["transcript_vtt"]).write_text("WEBVTT", encoding="utf-8")

    assert transcript_artifacts_ready(output_paths) is False

    Path(output_paths["transcript_tsv"]).write_text("0\thello", encoding="utf-8")
    assert transcript_artifacts_ready(output_paths) is True


def test_plan_frame_timestamps_prefers_dense_segments_across_full_video() -> None:
    timestamps = plan_frame_timestamps(
        duration_seconds=450.0,
        frame_interval=60,
        max_frames=4,
        transcript_segments=[
            {"start": 20.0, "end": 40.0, "text": "a" * 80},
            {"start": 130.0, "end": 170.0, "text": "b" * 120},
            {"start": 250.0, "end": 290.0, "text": "c" * 120},
            {"start": 360.0, "end": 410.0, "text": "d" * 140},
        ],
    )

    assert timestamps == [30, 150, 270, 385]


def test_plan_visual_supplemental_timestamps_targets_neighbors_of_weak_candidates() -> None:
    timestamps = plan_visual_supplemental_timestamps(
        duration_seconds=450.0,
        candidate_frames=[
            {"timestamp": 0.0, "ocr_text": "transformer BERTHY identity card", "is_low_value": False},
            {"timestamp": 37.0, "ocr_text": "WF", "is_low_value": True},
            {"timestamp": 74.0, "ocr_text": "WS", "is_low_value": True},
        ],
    )

    assert timestamps == [25, 49, 62, 86]


def test_annotate_scene_change_scores_marks_previous_stable_frame_as_page_change_candidate(tmp_path: Path) -> None:
    def write_frame(name: str, *, variant: str) -> Path:
        path = tmp_path / name
        image = Image.new("RGB", (320, 180), "white")
        if variant == "a":
            for x in range(20, 300):
                for y in range(30, 70):
                    image.putpixel((x, y), (40, 40, 40))
            for x in range(30, 180):
                for y in range(90, 150):
                    image.putpixel((x, y), (200, 200, 200))
        else:
            for x in range(40, 280):
                for y in range(20, 120):
                    image.putpixel((x, y), (20, 20, 20))
            for x in range(210, 300):
                for y in range(130, 165):
                    image.putpixel((x, y), (220, 220, 220))
        image.save(path)
        return path

    slide_a1 = write_frame("frame_001.jpg", variant="a")
    slide_a2 = write_frame("frame_002.jpg", variant="a")
    slide_a3 = write_frame("frame_003.jpg", variant="a")
    slide_b = write_frame("frame_004.jpg", variant="b")

    annotated = annotate_scene_change_scores(
        [
            {"timestamp": 0.0, "path": str(slide_a1), "ocr_text": "intro slide"},
            {"timestamp": 10.0, "path": str(slide_a2), "ocr_text": "intro slide"},
            {"timestamp": 20.0, "path": str(slide_a3), "ocr_text": "intro slide"},
            {"timestamp": 30.0, "path": str(slide_b), "ocr_text": "next slide"},
        ]
    )

    assert annotated[2]["page_change_candidate"] is True
    assert annotated[2]["page_change_reason"] == "stable_before_change"
    assert annotated[3]["change_kind"] == "page_change"


def test_annotate_scene_change_scores_filters_small_markup_changes(tmp_path: Path) -> None:
    base = tmp_path / "base.jpg"
    marked = tmp_path / "marked.jpg"
    image = Image.new("RGB", (320, 180), "white")
    for x in range(30, 290):
        for y in range(40, 140):
            image.putpixel((x, y), (240, 240, 240))
    image.save(base)

    marked_image = image.copy()
    for x in range(260, 280):
        for y in range(120, 140):
            marked_image.putpixel((x, y), (255, 0, 0))
    marked_image.save(marked)

    annotated = annotate_scene_change_scores(
        [
            {"timestamp": 0.0, "path": str(base), "ocr_text": "same slide"},
            {"timestamp": 10.0, "path": str(marked), "ocr_text": "same slide"},
        ]
    )

    assert annotated[1]["change_kind"] in {"no_change", "annotation_like_change"}
    assert annotated[1]["change_ratio"] < 0.02
    assert annotated[1]["page_change_candidate"] is True


def test_scan_visual_candidate_timestamps_does_not_backfill_uniform_steps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import video_to_notes_visual

    stable_a = np.full((180, 320), 20, dtype=np.uint8)
    stable_b = np.full((180, 320), 220, dtype=np.uint8)
    fake_frames = [stable_a.copy() for _ in range(5)] + [stable_b.copy() for _ in range(5)]

    class FakeCapture:
        def __init__(self, _path: str) -> None:
            self.current_index = 0

        def isOpened(self) -> bool:
            return True

        def get(self, prop: int) -> float:
            if prop == video_to_notes_visual.cv2.CAP_PROP_FPS:
                return 1.0
            if prop == video_to_notes_visual.cv2.CAP_PROP_FRAME_COUNT:
                return float(len(fake_frames))
            return 0.0

        def set(self, prop: int, value: float) -> bool:
            if prop == video_to_notes_visual.cv2.CAP_PROP_POS_FRAMES:
                self.current_index = int(value)
            return True

        def read(self):
            if 0 <= self.current_index < len(fake_frames):
                return True, fake_frames[self.current_index]
            return False, None

        def release(self) -> None:
            return None

    monkeypatch.setattr(video_to_notes_visual.cv2, "VideoCapture", FakeCapture)

    timestamps = scan_visual_candidate_timestamps(
        tmp_path / "demo.mp4",
        duration_seconds=10.0,
        max_candidates=18,
        sample_interval_seconds=1.0,
        min_gap_seconds=2,
    )

    assert timestamps == [0, 4, 9]


def test_merge_visual_candidates_deduplicates_by_timestamp() -> None:
    merged = merge_visual_candidates(
        [
            {"timestamp": 37.0, "relative_path": "visual_candidates/frame_002.jpg", "ocr_text": "WF"},
            {"timestamp": 74.0, "relative_path": "visual_candidates/frame_003.jpg", "ocr_text": "WS"},
        ],
        [
            {"timestamp": 37.0, "relative_path": "visual_candidates/frame_013.jpg", "ocr_text": "better"},
            {"timestamp": 49.0, "relative_path": "visual_candidates/frame_014.jpg", "ocr_text": "BERT CLS"},
        ],
    )

    assert [item["timestamp"] for item in merged] == [37.0, 49.0, 74.0]
    assert merged[0]["relative_path"] == "visual_candidates/frame_013.jpg"


def test_frames_artifacts_ready_requires_expected_frame_count(tmp_path: Path) -> None:
    output_paths = plan_output_paths(Path("/tmp/4.1BERT-2训练.mp4"), tmp_path)
    Path(output_paths["visual_candidates_dir"]).mkdir(parents=True)
    expected = expected_frame_count(duration_seconds=450.0, frame_interval=60, max_frames=4)
    for index in range(expected - 1):
        (Path(output_paths["visual_candidates_dir"]) / f"frame_{index + 1:03d}.jpg").write_bytes(b"jpg")

    assert frames_artifacts_ready(output_paths, duration_seconds=450.0, frame_interval=60, max_frames=4) is False

    (Path(output_paths["visual_candidates_dir"]) / f"frame_{expected:03d}.jpg").write_bytes(b"jpg")
    assert frames_artifacts_ready(output_paths, duration_seconds=450.0, frame_interval=60, max_frames=4) is True


def test_build_review_segments_extends_to_sentence_boundary_when_split_is_nearby(tmp_path: Path) -> None:
    transcript_txt = tmp_path / "transcript.txt"
    transcript_json = tmp_path / "transcript.json"
    transcript_txt.write_text("第一句前半部分 第一局后半部分。 第二句开始。", encoding="utf-8")
    transcript_json.write_text(
        json.dumps(
            {
                "segments": [
                    {"start": 0.0, "end": 10.0, "text": "第一句前半部分"},
                    {"start": 10.0, "end": 18.0, "text": "第一句后半部分。"},
                    {"start": 18.0, "end": 28.0, "text": "第二句开始。"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    payload = build_review_segments(
        transcript_txt=transcript_txt,
        transcript_json=transcript_json,
        frames=[],
        max_chars=8,
        max_duration=12.0,
    )

    assert payload["segment_count"] == 2
    assert payload["segments"][0]["text"].endswith("。")
    assert "第一句后半部分。" in payload["segments"][0]["text"]


def test_ocr_artifacts_ready_requires_frame_count_match(tmp_path: Path) -> None:
    output_paths = plan_output_paths(Path("/tmp/4.1BERT-2训练.mp4"), tmp_path)
    visual_candidates_dir = Path(output_paths["visual_candidates_dir"])
    video_to_notes.ensure_dirs(output_paths)
    for index in range(2):
        (visual_candidates_dir / f"frame_{index + 1:03d}.jpg").write_bytes(b"jpg")

    ocr_json = Path(output_paths["visual_candidates_ocr_json"])
    ocr_json.write_text(json.dumps([{"relative_path": "visual_candidates/frame_001.jpg"}]), encoding="utf-8")
    assert ocr_artifacts_ready(output_paths) is False

    ocr_json.write_text(
        json.dumps(
            [
                {"relative_path": "visual_candidates/frame_001.jpg", "ocr_text": "first"},
                {"relative_path": "visual_candidates/frame_002.jpg", "ocr_text": "second"},
            ]
        ),
        encoding="utf-8",
    )
    assert ocr_artifacts_ready(output_paths) is True


def test_review_artifacts_ready_requires_fixed_schema(tmp_path: Path) -> None:
    output_paths = plan_output_paths(Path("/tmp/4.1BERT-2训练.mp4"), tmp_path)
    work_dir = Path(output_paths["work_dir"])
    video_to_notes.ensure_dirs(output_paths)
    Path(output_paths["transcript_cleaned_txt"]).write_text("cleaned", encoding="utf-8")
    Path(output_paths["codex_review_prompt_md"]).write_text("prompt", encoding="utf-8")
    write_review_segments(Path(output_paths["review_segments_json"]))
    Path(output_paths["transcript_corrections_json"]).write_text(json.dumps({"review_status": "pending"}), encoding="utf-8")

    assert review_artifacts_ready(output_paths) is False

    Path(output_paths["transcript_corrections_json"]).write_text(
        json.dumps(
            {
                "review_status": "pending",
                "last_updated": None,
                "segment_reviews": [],
                "corrections": [],
            }
        ),
        encoding="utf-8",
    )
    assert review_artifacts_ready(output_paths) is True


def test_codex_review_ready_requires_passed_report_and_completed_artifacts(tmp_path: Path) -> None:
    output_paths = plan_output_paths(Path("/tmp/4.1BERT-2训练.mp4"), tmp_path)
    video_to_notes.ensure_dirs(output_paths)
    raw = Path(output_paths["transcript_txt"])
    cleaned = Path(output_paths["transcript_cleaned_txt"])
    corrections = Path(output_paths["transcript_corrections_json"])
    review_segments = Path(output_paths["review_segments_json"])
    raw.write_text("Bort " * 80, encoding="utf-8")
    cleaned.write_text("BERT " * 80, encoding="utf-8")
    Path(output_paths["codex_review_prompt_md"]).write_text("prompt", encoding="utf-8")
    write_review_segments(review_segments)
    corrections.write_text(
        json.dumps(
            {
                "source_transcript": "transcript.txt",
                "cleaned_transcript": "transcript.cleaned.txt",
                "review_status": "done",
                "last_updated": "2026-03-13T04:00:00+08:00",
                "segment_reviews": [
                    {
                        "segment_id": "segment_001",
                        "start": 0.0,
                        "end": 60.0,
                        "summary": "ok",
                        "issues": ["Bort -> BERT"],
                        "status": "done",
                    }
                ],
                "corrections": [
                    {"raw": "Bort", "cleaned": "BERT", "reason": "term", "evidence": ["context"]},
                    {"raw": "Bort2", "cleaned": "BERT2", "reason": "term", "evidence": ["context"]},
                    {"raw": "Bort3", "cleaned": "BERT3", "reason": "term", "evidence": ["context"]},
                ],
            }
        ),
        encoding="utf-8",
    )
    Path(output_paths["review_report_json"]).write_text(json.dumps({"status": "failed"}), encoding="utf-8")
    assert codex_review_ready(output_paths) is False

    Path(output_paths["review_report_json"]).write_text(json.dumps({"status": "passed"}), encoding="utf-8")
    assert codex_review_ready(output_paths) is True


def test_resolve_stage_plan_skips_completed_stages_by_default(tmp_path: Path) -> None:
    output_paths = plan_output_paths(Path("/tmp/4.1BERT-2训练.mp4"), tmp_path)
    work_dir = Path(output_paths["work_dir"])
    visual_candidates_dir = Path(output_paths["visual_candidates_dir"])
    video_to_notes.ensure_dirs(output_paths)
    Path(output_paths["audio_path"]).write_bytes(b"wav")
    Path(output_paths["transcript_txt"]).write_text("hello", encoding="utf-8")
    Path(output_paths["transcript_json"]).write_text(json.dumps({"segments": [{"start": 0.0, "end": 1.0, "text": "hello"}]}), encoding="utf-8")
    Path(output_paths["transcript_srt"]).write_text("1", encoding="utf-8")
    Path(output_paths["transcript_vtt"]).write_text("WEBVTT", encoding="utf-8")
    Path(output_paths["transcript_tsv"]).write_text("0\thello", encoding="utf-8")
    for index in range(expected_frame_count(duration_seconds=450.0, frame_interval=60, max_frames=4)):
        (visual_candidates_dir / f"frame_{index + 1:03d}.jpg").write_bytes(b"jpg")
    Path(output_paths["visual_candidates_ocr_json"]).write_text(
        json.dumps([{"relative_path": f"visual_candidates/frame_{index + 1:03d}.jpg", "ocr_text": "x"} for index in range(4)]),
        encoding="utf-8",
    )
    Path(output_paths["transcript_cleaned_txt"]).write_text("BERT " * 80, encoding="utf-8")
    write_review_segments(Path(output_paths["review_segments_json"]))
    Path(output_paths["codex_review_prompt_md"]).write_text("prompt", encoding="utf-8")
    Path(output_paths["transcript_corrections_json"]).write_text(
        json.dumps(
            {
                "source_transcript": "transcript.txt",
                "cleaned_transcript": "transcript.cleaned.txt",
                "review_status": "done",
                "last_updated": "2026-03-13T04:00:00+08:00",
                "segment_reviews": [
                    {"segment_id": "segment_001", "start": 0.0, "end": 60.0, "summary": "ok", "issues": [], "status": "done"}
                ],
                "corrections": [
                    {"raw": "Bort", "cleaned": "BERT", "reason": "term", "evidence": ["context"]},
                    {"raw": "Bort2", "cleaned": "BERT2", "reason": "term", "evidence": ["context"]},
                ],
            }
        ),
        encoding="utf-8",
    )
    Path(output_paths["review_report_json"]).write_text(json.dumps({"status": "passed"}), encoding="utf-8")
    Path(output_paths["note_path"]).write_text("# note", encoding="utf-8")
    Path(output_paths["note_generation_report_json"]).write_text(
        json.dumps({"status": "passed", "quality_gate_passed": True}),
        encoding="utf-8",
    )

    stage_plan = resolve_stage_plan(
        output_paths=output_paths,
        duration_seconds=450.0,
        frame_interval=60,
        max_frames=4,
        force={},
    )

    assert stage_plan["audio"]["run"] is False
    assert stage_plan["transcribe"]["run"] is False
    assert stage_plan["frames"]["run"] is False
    assert stage_plan["ocr"]["run"] is False
    assert stage_plan["review_artifacts"]["run"] is False
    assert stage_plan["codex_review"]["run"] is False
    assert stage_plan["note"]["run"] is False
    assert stage_plan["metadata"]["run"] is True


def test_resolve_stage_plan_force_transcribe_invalidates_downstream_stages(tmp_path: Path) -> None:
    output_paths = plan_output_paths(Path("/tmp/4.1BERT-2训练.mp4"), tmp_path)
    stage_plan = resolve_stage_plan(
        output_paths=output_paths,
        duration_seconds=450.0,
        frame_interval=60,
        max_frames=4,
        force={"transcribe": True},
    )

    assert stage_plan["audio"]["run"] is True
    assert stage_plan["transcribe"]["run"] is True
    assert stage_plan["frames"]["run"] is True
    assert stage_plan["ocr"]["run"] is True
    assert stage_plan["review_artifacts"]["run"] is True
    assert stage_plan["codex_review"]["run"] is True
    assert stage_plan["note"]["run"] is True
    assert stage_plan["metadata"]["run"] is True


def test_resolve_stage_plan_force_codex_review_only_reruns_review_and_note(tmp_path: Path) -> None:
    output_paths = plan_output_paths(Path("/tmp/4.1BERT-2训练.mp4"), tmp_path)
    stage_plan = resolve_stage_plan(
        output_paths=output_paths,
        duration_seconds=450.0,
        frame_interval=60,
        max_frames=4,
        force={"codex_review": True},
    )

    assert stage_plan["audio"]["run"] is True
    assert stage_plan["transcribe"]["run"] is True
    assert stage_plan["frames"]["run"] is True
    assert stage_plan["ocr"]["run"] is True
    assert stage_plan["review_artifacts"]["run"] is True
    assert stage_plan["codex_review"]["run"] is True
    assert stage_plan["note"]["run"] is True


def test_resolve_stage_plan_force_note_only_preserves_upstream_stages(tmp_path: Path) -> None:
    output_paths = plan_output_paths(Path("/tmp/1.1.1RNN基本原理.mp4"), tmp_path)
    work_dir = Path(output_paths["work_dir"])
    visual_candidates_dir = Path(output_paths["visual_candidates_dir"])
    video_to_notes.ensure_dirs(output_paths)
    Path(output_paths["audio_path"]).write_bytes(b"wav")
    Path(output_paths["transcript_txt"]).write_text("raw", encoding="utf-8")
    Path(output_paths["transcript_json"]).write_text(
        json.dumps({"segments": [{"start": 0.0, "end": 1.0, "text": "hello"}]}),
        encoding="utf-8",
    )
    Path(output_paths["transcript_srt"]).write_text("1", encoding="utf-8")
    Path(output_paths["transcript_vtt"]).write_text("WEBVTT", encoding="utf-8")
    Path(output_paths["transcript_tsv"]).write_text("0\thello", encoding="utf-8")
    for index in range(4):
        (visual_candidates_dir / f"frame_{index + 1:03d}.jpg").write_bytes(b"jpg")
    Path(output_paths["visual_candidates_ocr_json"]).write_text(
        json.dumps(
            [
                {"relative_path": f"visual_candidates/frame_{index + 1:03d}.jpg", "ocr_text": "x"}
                for index in range(4)
            ]
        ),
        encoding="utf-8",
    )
    Path(output_paths["transcript_cleaned_txt"]).write_text("cleaned", encoding="utf-8")
    Path(output_paths["codex_review_prompt_md"]).write_text("prompt", encoding="utf-8")
    write_review_segments(Path(output_paths["review_segments_json"]))
    Path(output_paths["transcript_corrections_json"]).write_text(
        json.dumps(
            {
                "review_status": "done",
                "last_updated": "2026-03-15T00:00:00+08:00",
                "segment_reviews": [
                    {"segment_id": "segment_001", "start": 0.0, "end": 60.0, "summary": "ok", "issues": [], "status": "done"}
                ],
                "corrections": [{"raw": "a", "cleaned": "b", "reason": "ok", "evidence": ["ctx"]}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    Path(output_paths["review_report_json"]).write_text(json.dumps({"status": "passed"}), encoding="utf-8")
    Path(output_paths["note_path"]).write_text(VALID_RULE_BASED_NOTE, encoding="utf-8")
    Path(output_paths["note_generation_report_json"]).write_text(
        json.dumps({"status": "passed", "quality_gate_passed": True}),
        encoding="utf-8",
    )

    stage_plan = resolve_stage_plan(
        output_paths=output_paths,
        duration_seconds=450.0,
        frame_interval=60,
        max_frames=4,
        force={"note": True},
    )

    assert stage_plan["audio"]["run"] is False
    assert stage_plan["transcribe"]["run"] is False
    assert stage_plan["frames"]["run"] is False
    assert stage_plan["ocr"]["run"] is False
    assert stage_plan["review_artifacts"]["run"] is False
    assert stage_plan["codex_review"]["run"] is False
    assert stage_plan["note"]["run"] is True
    assert stage_plan["metadata"]["run"] is True
    assert stage_plan["frames"]["reason"] == "preserved for note-only rerender"


def test_resolve_stage_plan_force_note_only_reruns_missing_upstream_stages(tmp_path: Path) -> None:
    output_paths = plan_output_paths(Path("/tmp/1.1.1RNN基本原理.mp4"), tmp_path)
    video_to_notes.ensure_dirs(output_paths)

    stage_plan = resolve_stage_plan(
        output_paths=output_paths,
        duration_seconds=450.0,
        frame_interval=60,
        max_frames=4,
        force={"note": True},
    )

    assert stage_plan["audio"]["run"] is True
    assert stage_plan["audio"]["reason"] == "artifacts missing or incomplete"
    assert stage_plan["transcribe"]["run"] is True
    assert stage_plan["frames"]["run"] is True
    assert stage_plan["ocr"]["run"] is True
    assert stage_plan["review_artifacts"]["run"] is True
    assert stage_plan["codex_review"]["run"] is True
    assert stage_plan["note"]["run"] is True


def test_run_pipeline_skips_completed_heavy_stages(tmp_path: Path, monkeypatch) -> None:
    input_video = tmp_path / "4.1BERT-2训练.mp4"
    input_video.write_bytes(b"mp4")
    output_paths = plan_output_paths(input_video, tmp_path)
    work_dir = Path(output_paths["work_dir"])
    visual_candidates_dir = Path(output_paths["visual_candidates_dir"])
    video_to_notes.ensure_dirs(output_paths)
    Path(output_paths["audio_path"]).write_bytes(b"wav")
    Path(output_paths["transcript_txt"]).write_text("Bort " * 80, encoding="utf-8")
    Path(output_paths["transcript_json"]).write_text(
        json.dumps({"segments": [{"start": 0.0, "end": 1.0, "text": "hello"}]}),
        encoding="utf-8",
    )
    Path(output_paths["transcript_srt"]).write_text("1", encoding="utf-8")
    Path(output_paths["transcript_vtt"]).write_text("WEBVTT", encoding="utf-8")
    Path(output_paths["transcript_tsv"]).write_text("0\thello", encoding="utf-8")
    for index in range(4):
        (visual_candidates_dir / f"frame_{index + 1:03d}.jpg").write_bytes(b"jpg")
    Path(output_paths["visual_candidates_ocr_json"]).write_text(
        json.dumps([{"relative_path": f"visual_candidates/frame_{index + 1:03d}.jpg", "ocr_text": "x"} for index in range(4)]),
        encoding="utf-8",
    )
    Path(output_paths["transcript_cleaned_txt"]).write_text("BERT " * 80, encoding="utf-8")
    Path(output_paths["codex_review_prompt_md"]).write_text("prompt", encoding="utf-8")
    write_review_segments(Path(output_paths["review_segments_json"]))
    Path(output_paths["transcript_corrections_json"]).write_text(
        json.dumps(
            {
                "source_transcript": "transcript.txt",
                "cleaned_transcript": "transcript.cleaned.txt",
                "review_status": "done",
                "last_updated": "2026-03-13T04:00:00+08:00",
                "segment_reviews": [
                    {"segment_id": "segment_001", "start": 0.0, "end": 60.0, "summary": "ok", "issues": [], "status": "done"}
                ],
                "corrections": [
                    {"raw": "Bort", "cleaned": "BERT", "reason": "term", "evidence": ["context"]},
                    {"raw": "Bort2", "cleaned": "BERT2", "reason": "term", "evidence": ["context"]},
                    {"raw": "Bort3", "cleaned": "BERT3", "reason": "term", "evidence": ["context"]},
                ],
            }
        ),
        encoding="utf-8",
    )
    Path(output_paths["review_report_json"]).write_text(json.dumps({"status": "passed"}), encoding="utf-8")
    Path(output_paths["note_path"]).write_text("# note", encoding="utf-8")
    Path(output_paths["note_generation_report_json"]).write_text(
        json.dumps({"status": "passed", "quality_gate_passed": True}),
        encoding="utf-8",
    )

    calls: list[str] = []

    def record(name: str):
        def inner(*args, **kwargs):
            calls.append(name)
            if name == "ffprobe_duration":
                return 450.0
            if name == "ocr_frames":
                return []
            return None
        return inner

    monkeypatch.setattr(video_to_notes_pipeline, "require_binary", record("require_binary"))
    monkeypatch.setattr(video_to_notes_pipeline, "ffprobe_duration", record("ffprobe_duration"))
    monkeypatch.setattr(video_to_notes_pipeline, "extract_audio", record("extract_audio"))
    monkeypatch.setattr(video_to_notes_pipeline, "transcribe_audio", record("transcribe_audio"))
    monkeypatch.setattr(video_to_notes_pipeline, "move_whisper_outputs", record("move_whisper_outputs"))
    monkeypatch.setattr(video_to_notes_pipeline, "scan_visual_candidate_timestamps", lambda *_args, **_kwargs: [0.0])
    monkeypatch.setattr(
        video_to_notes_pipeline,
        "extract_frames_at_timestamps",
        lambda *_args, **_kwargs: [{"timestamp": 0.0, "path": "frame", "relative_path": "visual_candidates/frame_001.jpg"}],
    )
    monkeypatch.setattr(video_to_notes_pipeline, "ocr_frames", record("ocr_frames"))
    monkeypatch.setattr(video_to_notes_pipeline, "ensure_review_artifacts", record("ensure_review_artifacts"))
    monkeypatch.setattr(video_to_notes_pipeline, "run_codex_review", record("run_codex_review"))
    monkeypatch.setattr(video_to_notes_pipeline, "write_metadata", record("write_metadata"))
    monkeypatch.setattr(video_to_notes_pipeline, "build_note_markdown", lambda **kwargs: VALID_RULE_BASED_NOTE)
    monkeypatch.setattr(video_to_notes_pipeline, "resolve_slides_path", lambda *_args, **_kwargs: None)

    args = SimpleNamespace(
        input=str(input_video),
        output_root=str(tmp_path),
        frame_interval=60,
        max_frames=4,
        whisper_model="small",
        language="zh",
        codex_timeout_seconds=300,
        codex_review_parallelism=2,
        skip_codex_review=False,
        force_audio=False,
        force_transcribe=False,
        force_frames=False,
        force_ocr=False,
        force_review_artifacts=False,
        force_codex_review=False,
        force_note=False,
    )

    result = video_to_notes_pipeline.run_pipeline(args)

    assert result["stage_plan"]["audio"]["run"] is False
    assert result["stage_plan"]["transcribe"]["run"] is False
    assert result["stage_plan"]["frames"]["run"] is False
    assert result["stage_plan"]["ocr"]["run"] is False
    assert result["stage_plan"]["codex_review"]["run"] is False
    assert "extract_audio" not in calls
    assert "transcribe_audio" not in calls
    assert "extract_frames_at_timestamps" not in calls
    assert "ocr_frames" not in calls
    assert "run_codex_review" not in calls
    assert "write_metadata" in calls
    visual_alignment = json.loads(Path(output_paths["visual_alignment_json"]).read_text(encoding="utf-8"))
    assert visual_alignment["segment_count"] == 1
    assert visual_alignment["segments"][0]["segment_id"] == "segment_001"
    assert visual_alignment["segments"][0]["has_useful_visual"] is True
    assert Path(output_paths["ppt_alignment_debug_md"]).exists()
    note_outline = json.loads(Path(output_paths["note_outline_json"]).read_text(encoding="utf-8"))
    note_blocks = json.loads(Path(output_paths["note_blocks_json"]).read_text(encoding="utf-8"))
    assert note_outline["chapter_count"] == 1
    assert note_blocks["block_count"] == 1


def test_run_pipeline_prints_stage_skip_and_run_logs(tmp_path: Path, monkeypatch, capsys) -> None:
    input_video = tmp_path / "4.1BERT-2训练.mp4"
    input_video.write_bytes(b"mp4")
    output_paths = plan_output_paths(input_video, tmp_path)
    work_dir = Path(output_paths["work_dir"])
    visual_candidates_dir = Path(output_paths["visual_candidates_dir"])
    video_to_notes.ensure_dirs(output_paths)
    Path(output_paths["audio_path"]).write_bytes(b"wav")
    Path(output_paths["transcript_txt"]).write_text("Bort " * 80, encoding="utf-8")
    Path(output_paths["transcript_json"]).write_text(
        json.dumps({"segments": [{"start": 0.0, "end": 1.0, "text": "hello"}]}),
        encoding="utf-8",
    )
    Path(output_paths["transcript_srt"]).write_text("1", encoding="utf-8")
    Path(output_paths["transcript_vtt"]).write_text("WEBVTT", encoding="utf-8")
    Path(output_paths["transcript_tsv"]).write_text("0\thello", encoding="utf-8")
    for index in range(4):
        (visual_candidates_dir / f"frame_{index + 1:03d}.jpg").write_bytes(b"jpg")
    Path(output_paths["visual_candidates_ocr_json"]).write_text(
        json.dumps([{"relative_path": f"visual_candidates/frame_{index + 1:03d}.jpg", "ocr_text": "x"} for index in range(4)]),
        encoding="utf-8",
    )
    Path(output_paths["transcript_cleaned_txt"]).write_text("BERT " * 80, encoding="utf-8")
    Path(output_paths["codex_review_prompt_md"]).write_text("prompt", encoding="utf-8")
    write_review_segments(Path(output_paths["review_segments_json"]))
    Path(output_paths["transcript_corrections_json"]).write_text(
        json.dumps(
            {
                "source_transcript": "transcript.txt",
                "cleaned_transcript": "transcript.cleaned.txt",
                "review_status": "done",
                "last_updated": "2026-03-13T04:00:00+08:00",
                "segment_reviews": [
                    {"segment_id": "segment_001", "start": 0.0, "end": 60.0, "summary": "ok", "issues": [], "status": "done"}
                ],
                "corrections": [
                    {"raw": "Bort", "cleaned": "BERT", "reason": "term", "evidence": ["context"]},
                    {"raw": "Bort2", "cleaned": "BERT2", "reason": "term", "evidence": ["context"]},
                    {"raw": "Bort3", "cleaned": "BERT3", "reason": "term", "evidence": ["context"]},
                ],
            }
        ),
        encoding="utf-8",
    )
    Path(output_paths["review_report_json"]).write_text(json.dumps({"status": "passed"}), encoding="utf-8")
    Path(output_paths["note_path"]).write_text("# note", encoding="utf-8")
    Path(output_paths["note_generation_report_json"]).write_text(
        json.dumps({"status": "passed", "quality_gate_passed": True}),
        encoding="utf-8",
    )

    monkeypatch.setattr(video_to_notes_pipeline, "require_binary", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(video_to_notes_pipeline, "ffprobe_duration", lambda *_args, **_kwargs: 450.0)
    monkeypatch.setattr(video_to_notes_pipeline, "extract_audio", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(video_to_notes_pipeline, "transcribe_audio", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(video_to_notes_pipeline, "move_whisper_outputs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(video_to_notes_pipeline, "scan_visual_candidate_timestamps", lambda *_args, **_kwargs: [0.0])
    monkeypatch.setattr(
        video_to_notes_pipeline,
        "extract_frames_at_timestamps",
        lambda *_args, **_kwargs: [{"timestamp": 0.0, "path": "frame", "relative_path": "visual_candidates/frame_001.jpg"}],
    )
    monkeypatch.setattr(video_to_notes_pipeline, "ocr_frames", lambda frames, *_args, **_kwargs: frames)
    monkeypatch.setattr(video_to_notes_pipeline, "ensure_review_artifacts", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(video_to_notes_pipeline, "run_codex_review", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(video_to_notes_pipeline, "write_metadata", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(video_to_notes_pipeline, "build_note_markdown", lambda **kwargs: VALID_RULE_BASED_NOTE)
    monkeypatch.setattr(video_to_notes_pipeline, "resolve_slides_path", lambda *_args, **_kwargs: None)

    args = SimpleNamespace(
        input=str(input_video),
        output_root=str(tmp_path),
        frame_interval=60,
        max_frames=4,
        whisper_model="small",
        language="zh",
        codex_timeout_seconds=300,
        codex_review_parallelism=2,
        skip_codex_review=False,
        force_audio=False,
        force_transcribe=False,
        force_frames=False,
        force_ocr=False,
        force_review_artifacts=False,
        force_codex_review=False,
        force_note=False,
    )

    video_to_notes_pipeline.run_pipeline(args)
    stdout = capsys.readouterr().out

    assert "SKIP audio" in stdout
    assert "SKIP transcribe" in stdout
    assert "SKIP codex_review" in stdout
    assert "RUN metadata" in stdout
    assert "artifacts already complete" in stdout


def test_run_pipeline_force_transcribe_reruns_transcribe_and_downstream(tmp_path: Path, monkeypatch) -> None:
    input_video = tmp_path / "4.1BERT-2训练.mp4"
    input_video.write_bytes(b"mp4")
    calls: list[str] = []

    def record(name: str):
        def inner(*args, **kwargs):
            calls.append(name)
            if name == "ffprobe_duration":
                return 450.0
            if name == "extract_frames_at_timestamps":
                return [{"timestamp": 0.0, "path": "frame", "relative_path": "visual_candidates/frame_001.jpg"}]
            if name == "ocr_frames":
                return [{"timestamp": 0.0, "path": "frame", "relative_path": "visual_candidates/frame_001.jpg", "ocr_text": "x"}]
            return None
        return inner

    monkeypatch.setattr(video_to_notes_pipeline, "require_binary", record("require_binary"))
    monkeypatch.setattr(video_to_notes_pipeline, "ffprobe_duration", record("ffprobe_duration"))
    monkeypatch.setattr(video_to_notes_pipeline, "extract_audio", record("extract_audio"))
    monkeypatch.setattr(video_to_notes_pipeline, "transcribe_audio", record("transcribe_audio"))
    monkeypatch.setattr(video_to_notes_pipeline, "move_whisper_outputs", record("move_whisper_outputs"))
    monkeypatch.setattr(video_to_notes_pipeline, "scan_visual_candidate_timestamps", lambda *_args, **_kwargs: [0])
    monkeypatch.setattr(video_to_notes_pipeline, "extract_frames_at_timestamps", record("extract_frames_at_timestamps"))
    monkeypatch.setattr(video_to_notes_pipeline, "ocr_frames", record("ocr_frames"))
    monkeypatch.setattr(video_to_notes_pipeline, "ensure_review_artifacts", record("ensure_review_artifacts"))
    monkeypatch.setattr(video_to_notes_pipeline, "run_codex_review", record("run_codex_review"))
    monkeypatch.setattr(video_to_notes_pipeline, "write_metadata", record("write_metadata"))
    monkeypatch.setattr(video_to_notes_pipeline, "build_note_markdown", lambda **kwargs: VALID_RULE_BASED_NOTE)
    monkeypatch.setattr(video_to_notes_pipeline, "read_excerpt", lambda *args, **kwargs: ["x"])
    monkeypatch.setattr(video_to_notes_pipeline, "resolve_slides_path", lambda *_args, **_kwargs: None)

    args = SimpleNamespace(
        input=str(input_video),
        output_root=str(tmp_path),
        frame_interval=60,
        max_frames=4,
        whisper_model="small",
        language="zh",
        codex_timeout_seconds=300,
        codex_review_parallelism=2,
        skip_codex_review=False,
        force_audio=False,
        force_transcribe=True,
        force_frames=False,
        force_ocr=False,
        force_review_artifacts=False,
        force_codex_review=False,
        force_note=False,
    )

    result = video_to_notes_pipeline.run_pipeline(args)

    assert result["stage_plan"]["transcribe"]["run"] is True
    assert "extract_audio" in calls
    assert "transcribe_audio" in calls
    assert "extract_frames_at_timestamps" in calls
    assert "ocr_frames" in calls
    assert "ensure_review_artifacts" in calls
    assert "run_codex_review" in calls
    assert "write_metadata" in calls


def test_run_pipeline_uses_codex_note_generation_stage_when_not_skipped(tmp_path: Path, monkeypatch) -> None:
    input_video = tmp_path / "4.1BERT-2训练.mp4"
    input_video.write_bytes(b"mp4")
    calls: list[str] = []

    def record(name: str):
        def inner(*args, **kwargs):
            calls.append(name)
            if name == "ffprobe_duration":
                return 450.0
            if name == "ocr_frames":
                return [{"timestamp": 0.0, "path": "frame", "relative_path": "assets/frame_001.jpg", "ocr_text": "x"}]
            if name == "run_codex_note_generation":
                output_paths = args[0]
                Path(output_paths["note_path"]).write_text("# codex note", encoding="utf-8")
                Path(output_paths["note_generation_report_json"]).write_text(
                    json.dumps({"status": "passed", "quality_gate_passed": True}),
                    encoding="utf-8",
                )
                return None
            return None
        return inner

    monkeypatch.setattr(video_to_notes_pipeline, "require_binary", record("require_binary"))
    monkeypatch.setattr(video_to_notes_pipeline, "ffprobe_duration", record("ffprobe_duration"))
    monkeypatch.setattr(video_to_notes_pipeline, "extract_audio", record("extract_audio"))
    monkeypatch.setattr(video_to_notes_pipeline, "transcribe_audio", record("transcribe_audio"))
    monkeypatch.setattr(video_to_notes_pipeline, "move_whisper_outputs", record("move_whisper_outputs"))
    monkeypatch.setattr(video_to_notes_pipeline, "scan_visual_candidate_timestamps", lambda *_args, **_kwargs: [0.0])
    monkeypatch.setattr(
        video_to_notes_pipeline,
        "extract_frames_at_timestamps",
        lambda *_args, **_kwargs: [{"timestamp": 0.0, "path": "frame", "relative_path": "visual_candidates/frame_001.jpg"}],
    )
    monkeypatch.setattr(video_to_notes_pipeline, "ocr_frames", record("ocr_frames"))
    monkeypatch.setattr(video_to_notes_pipeline, "ensure_review_artifacts", record("ensure_review_artifacts"))
    monkeypatch.setattr(video_to_notes_pipeline, "run_codex_review", record("run_codex_review"))
    monkeypatch.setattr(video_to_notes_pipeline, "run_codex_note_generation", record("run_codex_note_generation"))
    monkeypatch.setattr(video_to_notes_pipeline, "write_metadata", record("write_metadata"))
    monkeypatch.setattr(video_to_notes_pipeline, "read_excerpt", lambda *args, **kwargs: ["x"])
    monkeypatch.setattr(video_to_notes_pipeline, "resolve_slides_path", lambda *_args, **_kwargs: None)

    args = SimpleNamespace(
        input=str(input_video),
        output_root=str(tmp_path),
        frame_interval=60,
        max_frames=4,
        whisper_model="small",
        language="zh",
        codex_timeout_seconds=300,
        codex_review_parallelism=2,
        skip_codex_review=True,
        skip_codex_note=False,
        force_audio=False,
        force_transcribe=True,
        force_frames=False,
        force_ocr=False,
        force_review_artifacts=False,
        force_codex_review=False,
        force_note=False,
    )

    video_to_notes_pipeline.run_pipeline(args)

    assert "run_codex_note_generation" in calls


def test_run_pipeline_force_ocr_reuses_existing_frame_files(tmp_path: Path, monkeypatch) -> None:
    input_video = tmp_path / "4.1BERT-2训练.mp4"
    input_video.write_bytes(b"mp4")
    output_paths = plan_output_paths(input_video, tmp_path)
    work_dir = Path(output_paths["work_dir"])
    visual_candidates_dir = Path(output_paths["visual_candidates_dir"])
    video_to_notes.ensure_dirs(output_paths)
    Path(output_paths["audio_path"]).write_bytes(b"wav")
    Path(output_paths["transcript_txt"]).write_text("hello", encoding="utf-8")
    Path(output_paths["transcript_json"]).write_text(
        json.dumps({"segments": [{"start": 0.0, "end": 1.0, "text": "hello"}]}),
        encoding="utf-8",
    )
    Path(output_paths["transcript_srt"]).write_text("1", encoding="utf-8")
    Path(output_paths["transcript_vtt"]).write_text("WEBVTT", encoding="utf-8")
    Path(output_paths["transcript_tsv"]).write_text("0\thello", encoding="utf-8")
    for index in range(4):
        (visual_candidates_dir / f"frame_{index + 1:03d}.jpg").write_bytes(b"jpg")

    seen_frame_counts: list[int] = []

    def fake_ocr_frames(frames, *_args, **_kwargs):
        seen_frame_counts.append(len(frames))
        return [{"timestamp": 0.0, "path": frames[0]["path"], "relative_path": frames[0]["relative_path"], "ocr_text": "x"}]

    monkeypatch.setattr(video_to_notes_pipeline, "require_binary", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(video_to_notes_pipeline, "ffprobe_duration", lambda *_args, **_kwargs: 450.0)
    monkeypatch.setattr(video_to_notes_pipeline, "extract_audio", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(video_to_notes_pipeline, "transcribe_audio", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(video_to_notes_pipeline, "move_whisper_outputs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(video_to_notes_pipeline, "scan_visual_candidate_timestamps", lambda *_args, **_kwargs: [0.0])
    monkeypatch.setattr(
        video_to_notes_pipeline,
        "extract_frames_at_timestamps",
        lambda *_args, **_kwargs: [{"timestamp": 0.0, "path": "frame", "relative_path": "visual_candidates/frame_001.jpg"}],
    )
    monkeypatch.setattr(video_to_notes_pipeline, "ocr_frames", fake_ocr_frames)
    monkeypatch.setattr(video_to_notes_pipeline, "ensure_review_artifacts", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(video_to_notes_pipeline, "run_codex_review", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(video_to_notes_pipeline, "write_metadata", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(video_to_notes_pipeline, "build_note_markdown", lambda **kwargs: VALID_RULE_BASED_NOTE)
    monkeypatch.setattr(video_to_notes_pipeline, "read_excerpt", lambda *args, **kwargs: ["x"])
    monkeypatch.setattr(video_to_notes_pipeline, "resolve_slides_path", lambda *_args, **_kwargs: None)

    args = SimpleNamespace(
        input=str(input_video),
        output_root=str(tmp_path),
        frame_interval=60,
        max_frames=4,
        whisper_model="small",
        language="zh",
        codex_timeout_seconds=300,
        codex_review_parallelism=2,
        skip_codex_review=True,
        force_audio=False,
        force_transcribe=False,
        force_frames=False,
        force_ocr=True,
        force_review_artifacts=False,
        force_codex_review=False,
        force_note=False,
    )

    video_to_notes_pipeline.run_pipeline(args)

    assert seen_frame_counts[0] == 4
    assert seen_frame_counts[1:] == [1]


def test_run_pipeline_force_note_only_does_not_rerun_upstream_stages(tmp_path: Path, monkeypatch) -> None:
    input_video = tmp_path / "1.1.1RNN基本原理.mp4"
    input_video.write_bytes(b"mp4")
    output_paths = plan_output_paths(input_video, tmp_path)
    work_dir = Path(output_paths["work_dir"])
    visual_candidates_dir = Path(output_paths["visual_candidates_dir"])
    video_to_notes.ensure_dirs(output_paths)
    Path(output_paths["audio_path"]).write_bytes(b"wav")
    Path(output_paths["transcript_txt"]).write_text("raw transcript", encoding="utf-8")
    Path(output_paths["transcript_json"]).write_text(
        json.dumps({"segments": [{"start": 0.0, "end": 1.0, "text": "hello"}]}),
        encoding="utf-8",
    )
    Path(output_paths["transcript_srt"]).write_text("1", encoding="utf-8")
    Path(output_paths["transcript_vtt"]).write_text("WEBVTT", encoding="utf-8")
    Path(output_paths["transcript_tsv"]).write_text("0\thello", encoding="utf-8")
    Path(output_paths["transcript_cleaned_txt"]).write_text("cleaned transcript", encoding="utf-8")
    for index in range(4):
        (visual_candidates_dir / f"frame_{index + 1:03d}.jpg").write_bytes(b"jpg")
    Path(output_paths["visual_candidates_ocr_json"]).write_text(
        json.dumps(
            [{"relative_path": f"visual_candidates/frame_{index + 1:03d}.jpg", "ocr_text": "x"} for index in range(4)]
        ),
        encoding="utf-8",
    )
    Path(output_paths["review_segments_json"]).write_text(
        json.dumps(
            {
                "segment_count": 1,
                "segments": [
                    {
                        "segment_id": "segment_001",
                        "start": 0.0,
                        "end": 60.0,
                        "label": "00:00:00-00:01:00",
                        "text": "RNN 是循环神经网络",
                        "ocr_hints": [],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    Path(output_paths["codex_review_prompt_md"]).write_text("prompt", encoding="utf-8")
    Path(output_paths["transcript_corrections_json"]).write_text(
        json.dumps(
            {
                "review_status": "done",
                "last_updated": "2026-03-14T00:00:00+08:00",
                "segment_reviews": [
                    {
                        "segment_id": "segment_001",
                        "start": 0.0,
                        "end": 60.0,
                        "summary": "讲解 RNN 的基本概念",
                        "issues": [],
                        "status": "done",
                    }
                ],
                "corrections": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    Path(output_paths["review_report_json"]).write_text(json.dumps({"status": "passed"}), encoding="utf-8")
    Path(output_paths["visual_units_json"]).write_text(json.dumps({"unit_count": 0, "units": []}), encoding="utf-8")
    Path(output_paths["visual_alignment_json"]).write_text(
        json.dumps(
            {
                "segment_count": 1,
                "segments": [
                    {
                        "segment_id": "segment_001",
                        "start": 0.0,
                        "end": 60.0,
                        "summary": "讲解 RNN 的基本概念",
                        "cleaned_text": "RNN 通过隐藏状态保留时序信息。",
                        "raw_text": "raw",
                        "issues": [],
                        "selected_visuals": [],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    Path(output_paths["note_outline_json"]).write_text(
        json.dumps(
            {
                "chapter_count": 1,
                "chapters": [
                    {
                        "chapter_id": "chapter_001",
                        "title": "RNN基本原理",
                        "time_label": "00:00:00",
                        "sections": [
                            {
                                "section_id": "section_001",
                                "segment_id": "segment_001",
                                "title": "RNN 的基本概念",
                                "time_label": "00:00:00",
                                "start": 0.0,
                                "end": 60.0,
                            }
                        ],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    Path(output_paths["note_blocks_json"]).write_text(
        json.dumps(
            {
                "block_count": 1,
                "blocks": [
                    {
                        "section_id": "section_001",
                        "segment_id": "segment_001",
                        "title": "RNN 的基本概念",
                        "timestamp_ref": "[00:00]",
                        "summary": "讲解 RNN 的基本概念",
                        "definitions": ["RNN 是循环神经网络。"],
                        "key_points": ["RNN 通过隐藏状态保留时序信息。"],
                        "explanation_points": ["每个时刻同时接收当前输入和上一时刻状态。"],
                        "formula_candidates": [],
                        "visuals": [],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    Path(output_paths["slides_index_json"]).write_text(
        json.dumps(
            {
                "usable": True,
                "effective_visual_source_mode": "slides-first",
                "slides": [
                    {
                        "slide_id": "slide_001",
                        "slide_index": 1,
                        "title": "stale slide",
                        "text": "stale slide text",
                        "relative_path": "slides_preview/rendered/slide_001.png",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    calls: list[str] = []
    metadata_calls: list[dict[str, object]] = []

    def record(name: str):
        def inner(*_args, **_kwargs):
            calls.append(name)
            if name == "ffprobe_duration":
                return 450.0
            if name == "build_note_markdown":
                return VALID_RULE_BASED_NOTE
            return None
        return inner

    def record_metadata(*_args, **kwargs):
        calls.append("write_metadata")
        metadata_calls.append(kwargs)
        return None

    monkeypatch.setattr(video_to_notes_pipeline, "require_binary", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(video_to_notes_pipeline, "ffprobe_duration", record("ffprobe_duration"))
    monkeypatch.setattr(video_to_notes_pipeline, "extract_audio", record("extract_audio"))
    monkeypatch.setattr(video_to_notes_pipeline, "transcribe_audio", record("transcribe_audio"))
    monkeypatch.setattr(video_to_notes_pipeline, "move_whisper_outputs", record("move_whisper_outputs"))
    monkeypatch.setattr(video_to_notes_pipeline, "extract_frames_at_timestamps", record("extract_frames_at_timestamps"))
    monkeypatch.setattr(video_to_notes_pipeline, "scan_visual_candidate_timestamps", lambda *_args, **_kwargs: [0.0])
    monkeypatch.setattr(video_to_notes_pipeline, "ocr_frames", record("ocr_frames"))
    monkeypatch.setattr(video_to_notes_pipeline, "ensure_review_artifacts", record("ensure_review_artifacts"))
    monkeypatch.setattr(video_to_notes_pipeline, "run_codex_review", record("run_codex_review"))
    monkeypatch.setattr(video_to_notes_pipeline, "run_codex_note_generation", record("run_codex_note_generation"))
    monkeypatch.setattr(video_to_notes_pipeline, "build_note_markdown", record("build_note_markdown"))
    monkeypatch.setattr(video_to_notes_pipeline, "write_metadata", record_metadata)
    monkeypatch.setattr(video_to_notes_pipeline, "read_excerpt", lambda *args, **kwargs: ["x"])
    monkeypatch.setattr(video_to_notes_pipeline, "resolve_slides_path", lambda *_args, **_kwargs: None)

    args = SimpleNamespace(
        input=str(input_video),
        output_root=str(tmp_path),
        frame_interval=999,
        max_frames=99,
        whisper_model="large-v3",
        language="en",
        slides=None,
        codex_timeout_seconds=300,
        codex_review_parallelism=5,
        skip_codex_review=True,
        skip_codex_note=True,
        force_audio=False,
        force_transcribe=False,
        force_frames=False,
        force_ocr=False,
        force_review_artifacts=False,
        force_codex_review=False,
        force_note=True,
    )

    video_to_notes_pipeline.run_pipeline(args)

    assert "extract_audio" not in calls
    assert "transcribe_audio" not in calls
    assert "extract_frames_at_timestamps" not in calls
    assert "ocr_frames" not in calls
    assert "ensure_review_artifacts" not in calls
    assert "run_codex_review" not in calls
    assert "build_note_markdown" in calls
    visual_alignment = json.loads(Path(output_paths["visual_alignment_json"]).read_text(encoding="utf-8"))
    assert visual_alignment["visual_source_mode"] == "video-first"
    assert metadata_calls[-1]["visual_source_mode"] == "video-first"
    assert metadata_calls[-1]["slides_path"] is None
    assert metadata_calls[-1]["slides_usable"] is False


def test_preferred_transcript_path_uses_cleaned_when_available(tmp_path: Path) -> None:
    raw = tmp_path / "transcript.txt"
    cleaned = tmp_path / "transcript.cleaned.txt"
    raw.write_text("raw transcript", encoding="utf-8")
    cleaned.write_text("cleaned transcript", encoding="utf-8")

    assert preferred_transcript_path(raw, cleaned) == cleaned


def test_ensure_review_artifacts_creates_review_files_and_preserves_cleaned(tmp_path: Path) -> None:
    raw = tmp_path / "transcript.txt"
    cleaned = tmp_path / "transcript.cleaned.txt"
    corrections = tmp_path / "transcript.corrections.json"
    prompt = tmp_path / "codex_review_prompt.md"
    work_dir_agents = tmp_path / "AGENTS.md"
    transcript_json = tmp_path / "transcript.json"
    review_segments = tmp_path / "review_segments.json"
    raw.write_text("Bort 是一个研磨的语言模型", encoding="utf-8")
    cleaned.write_text("BERT 是一个预训练语言模型", encoding="utf-8")
    transcript_json.write_text(
        json.dumps(
            {
                "segments": [
                    {"start": 0.0, "end": 20.0, "text": "Bort 是一个研磨的语言模型"},
                    {"start": 20.0, "end": 40.0, "text": "这里介绍 BERT 预训练"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    ensure_review_artifacts(
        source_video=Path("materials/videos/4.1BERT-2训练.mp4"),
        transcript_txt=raw,
        transcript_cleaned_txt=cleaned,
        transcript_corrections_json=corrections,
        codex_review_prompt_md=prompt,
        work_dir_agents_md=work_dir_agents,
        transcript_json=transcript_json,
        review_segments_json=review_segments,
        frames=[
            {"timestamp": 60.0, "relative_path": "assets/frame_001.jpg", "ocr_text": "BERT 预训练"}
        ],
    )

    assert cleaned.read_text(encoding="utf-8") == "BERT 是一个预训练语言模型"
    assert corrections.exists()
    corrections_payload = corrections.read_text(encoding="utf-8")
    assert '"review_status": "pending"' in corrections_payload
    assert '"last_updated": null' in corrections_payload
    assert '"segment_reviews"' in corrections_payload
    assert '"segment_id": "segment_001"' in corrections_payload
    assert '"raw": ""' in corrections_payload
    assert '"cleaned": ""' in corrections_payload
    assert '"reason": ""' in corrections_payload
    assert '"evidence": []' in corrections_payload
    assert review_segments.exists()
    review_segments_payload = json.loads(review_segments.read_text(encoding="utf-8"))
    assert review_segments_payload["segment_count"] >= 1
    assert review_segments_payload["segments"][0]["segment_id"] == "segment_001"
    agent_text = work_dir_agents.read_text(encoding="utf-8")
    assert "当前目录是一个单视频审阅工作目录" in agent_text
    assert "不要运行 `video_to_notes.py`" in agent_text
    prompt_text = prompt.read_text(encoding="utf-8")
    assert "## Inputs" in prompt_text
    assert "## Outputs" in prompt_text
    assert "## Steps" in prompt_text
    assert "## Acceptance Criteria" in prompt_text
    assert "## Review Segments" in prompt_text
    assert "只修改 `transcript.cleaned.txt` 和 `transcript.corrections.json`" in prompt_text
    assert '"raw": "原错误词"' in prompt_text
    assert '"segment_id": "segment_001"' in prompt_text
    assert "transcript.txt" in prompt_text
    assert "review_segments.json" in prompt_text
    assert "transcript.cleaned.txt" in prompt_text
    assert "BERT 预训练" in prompt_text


def test_ensure_review_artifacts_migrates_existing_corrections_schema(tmp_path: Path) -> None:
    raw = tmp_path / "transcript.txt"
    cleaned = tmp_path / "transcript.cleaned.txt"
    corrections = tmp_path / "transcript.corrections.json"
    prompt = tmp_path / "codex_review_prompt.md"
    work_dir_agents = tmp_path / "AGENTS.md"
    transcript_json = tmp_path / "transcript.json"
    review_segments = tmp_path / "review_segments.json"
    raw.write_text("Bort 是一个研磨的语言模型", encoding="utf-8")
    cleaned.write_text("BERT 是一个预训练语言模型", encoding="utf-8")
    transcript_json.write_text(
        json.dumps({"segments": [{"start": 0.0, "end": 20.0, "text": "Bort 是一个研磨的语言模型"}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    corrections.write_text(
        '{"source_transcript":"transcript.txt","cleaned_transcript":"transcript.cleaned.txt","corrections":[{"raw":"Bort","cleaned":"BERT"}]}',
        encoding="utf-8",
    )

    ensure_review_artifacts(
        source_video=Path("materials/videos/4.1BERT-2训练.mp4"),
        transcript_txt=raw,
        transcript_cleaned_txt=cleaned,
        transcript_corrections_json=corrections,
        codex_review_prompt_md=prompt,
        work_dir_agents_md=work_dir_agents,
        transcript_json=transcript_json,
        review_segments_json=review_segments,
        frames=[],
    )

    migrated = corrections.read_text(encoding="utf-8")
    assert '"review_status": "pending"' in migrated
    assert '"last_updated": null' in migrated
    assert '"segment_reviews"' in migrated
    assert '"raw": "Bort"' in migrated
    assert '"cleaned": "BERT"' in migrated


def test_build_codex_exec_prompt_restricts_writable_files(tmp_path: Path) -> None:
    output_paths = plan_output_paths(Path("/tmp/4.1BERT-2训练.mp4"), tmp_path)

    prompt = build_codex_exec_prompt(output_paths)

    assert "codex_review_prompt.md" in prompt
    assert "transcript.txt" in prompt
    assert "review_segments.json" in prompt
    assert "transcript.cleaned.txt" in prompt
    assert "transcript.corrections.json" in prompt
    assert "不要修改 transcript.txt" in prompt
    assert "segment_reviews" in prompt
    assert "review_status" in prompt
    assert "不能与 `transcript.txt` 完全相同" in prompt
    assert "至少记录" in prompt


def test_build_codex_retry_prompt_includes_failure_feedback(tmp_path: Path) -> None:
    output_paths = plan_output_paths(Path("/tmp/4.1BERT-2训练.mp4"), tmp_path)

    prompt = build_codex_retry_prompt(
        output_paths,
        failure_reasons=[
            "transcript.cleaned.txt 与 transcript.txt 完全相同",
            "有效纠错条目不足 3 条",
        ],
    )

    assert "上一次审阅结果未通过质量校验" in prompt
    assert "transcript.cleaned.txt 与 transcript.txt 完全相同" in prompt
    assert "有效纠错条目不足 3 条" in prompt
    assert "请直接覆盖修正这两个文件" in prompt


def test_run_codex_note_generation_writes_passed_report(tmp_path: Path, monkeypatch) -> None:
    output_paths = plan_output_paths(Path("/tmp/4.1BERT-2训练.mp4"), tmp_path)
    work_dir = Path(output_paths["work_dir"])
    video_to_notes.ensure_dirs(output_paths)
    Path(output_paths["note_generation_prompt_md"]).write_text("prompt", encoding="utf-8")
    Path(output_paths["note_outline_json"]).write_text("{}", encoding="utf-8")
    Path(output_paths["note_blocks_json"]).write_text("{}", encoding="utf-8")

    def fake_run_command(
        command: list[str], *, cwd: Path | None = None, timeout: int | None = None
    ) -> subprocess.CompletedProcess[str]:
        assert cwd == work_dir
        Path(output_paths["note_path"]).write_text(
            "# 讲义\n\n## 知识小结\n\n核心定义卡片\n\n知识框架\n\n[00:58]\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "ok", "")

    monkeypatch.setattr(video_to_notes, "run_command", fake_run_command)

    run_codex_note_generation(output_paths, timeout_seconds=30)

    report = json.loads(Path(output_paths["note_generation_report_json"]).read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["generator"] == "codex"
    assert report["attempt_count"] == 1
    assert report["quality_gate_passed"] is True
    assert report["final_failure_reasons"] == []
    assert report["attempts"][0]["command_status"] == "completed"


def test_run_codex_note_generation_writes_failed_report_after_retry(tmp_path: Path, monkeypatch) -> None:
    output_paths = plan_output_paths(Path("/tmp/4.1BERT-2训练.mp4"), tmp_path)
    work_dir = Path(output_paths["work_dir"])
    video_to_notes.ensure_dirs(output_paths)
    Path(output_paths["note_generation_prompt_md"]).write_text("prompt", encoding="utf-8")
    Path(output_paths["note_outline_json"]).write_text("{}", encoding="utf-8")
    Path(output_paths["note_blocks_json"]).write_text("{}", encoding="utf-8")

    def fake_run_command(
        command: list[str], *, cwd: Path | None = None, timeout: int | None = None
    ) -> subprocess.CompletedProcess[str]:
        assert cwd == work_dir
        Path(output_paths["note_path"]).write_text("# bad note", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "not good", "")

    monkeypatch.setattr(video_to_notes, "run_command", fake_run_command)

    with pytest.raises(RuntimeError, match="note-generation failed after retry"):
        run_codex_note_generation(output_paths, timeout_seconds=30)

    report = json.loads(Path(output_paths["note_generation_report_json"]).read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["generator"] == "codex"
    assert report["attempt_count"] == 2
    assert report["quality_gate_passed"] is False
    assert "note.md 缺少知识小结" in report["final_failure_reasons"]
    assert report["attempts"][0]["command_status"] == "quality_gate_failed"
    assert report["attempts"][1]["command_status"] == "quality_gate_failed"


def test_run_codex_slides_cleanup_writes_passed_report(tmp_path: Path, monkeypatch) -> None:
    output_paths = plan_output_paths(Path("/tmp/1.1.1RNN基本原理.mp4"), tmp_path)
    work_dir = Path(output_paths["work_dir"])
    video_to_notes.ensure_dirs(output_paths)
    raw_payload = {
        "slide_count": 1,
        "slides": [
            {
                "slide_id": "slide_007",
                "slide_index": 7,
                "title": "RNN整体网络结构 ǜ֩−1 ǜ֩ ǜ֩+1",
                "text": "RNN整体网络结构 ǜ֩−1 ǜ֩ ǜ֩+1 我 喜欢 打 篮球 ǜ֩+2",
                "relative_path": "slides_preview/rendered/slide_007.png",
                "image_area": 1,
                "image_frequency": 1,
                "is_low_value": False,
            }
        ],
    }
    Path(output_paths["slides_index_raw_json"]).write_text(json.dumps(raw_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(output_paths["slides_index_json"]).write_text(json.dumps(sanitize_slides_payload(raw_payload), ensure_ascii=False, indent=2), encoding="utf-8")

    def fake_run_command(
        command: list[str], *, cwd: Path | None = None, timeout: int | None = None
    ) -> subprocess.CompletedProcess[str]:
        assert cwd == work_dir
        Path(output_paths["slides_index_json"]).write_text(
            json.dumps(
                {
                    "slide_count": 1,
                    "slides": [
                        {
                            "slide_id": "slide_007",
                            "slide_index": 7,
                            "title": "RNN整体网络结构",
                            "text": "RNN整体网络结构 我 喜欢 打 篮球",
                            "relative_path": "slides_preview/rendered/slide_007.png",
                            "image_area": 1,
                            "image_frequency": 1,
                            "is_low_value": False,
                        }
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "ok", "")

    monkeypatch.setattr(video_to_notes, "run_command", fake_run_command)

    cleaned = run_codex_slides_cleanup(output_paths, raw_payload=raw_payload, timeout_seconds=30)

    report = json.loads(Path(output_paths["slides_cleanup_report_json"]).read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["generator"] == "codex"
    assert report["attempt_count"] == 1
    assert report["noise_detected_before"] is True
    assert report["noise_detected_after"] is False
    assert cleaned["slides"][0]["title"] == "RNN整体网络结构"


def test_run_codex_slides_cleanup_accepts_timeout_with_valid_cleanup(tmp_path: Path, monkeypatch) -> None:
    output_paths = plan_output_paths(Path("/tmp/1.1.1RNN基本原理.mp4"), tmp_path)
    work_dir = Path(output_paths["work_dir"])
    video_to_notes.ensure_dirs(output_paths)
    raw_payload = {
        "slide_count": 1,
        "slides": [
            {
                "slide_id": "slide_007",
                "slide_index": 7,
                "title": "RNN整体网络结构 ǜ֩−1 ǜ֩ ǜ֩+1",
                "text": "RNN整体网络结构 ǜ֩−1 ǜ֩ ǜ֩+1 我 喜欢 打 篮球 ǜ֩+2",
                "relative_path": "slides_preview/rendered/slide_007.png",
                "image_area": 1,
                "image_frequency": 1,
                "is_low_value": False,
            }
        ],
    }
    Path(output_paths["slides_index_raw_json"]).write_text(json.dumps(raw_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(output_paths["slides_index_json"]).write_text(json.dumps(sanitize_slides_payload(raw_payload), ensure_ascii=False, indent=2), encoding="utf-8")

    def fake_run_command(
        command: list[str], *, cwd: Path | None = None, timeout: int | None = None
    ) -> subprocess.CompletedProcess[str]:
        assert cwd == work_dir
        Path(output_paths["slides_index_json"]).write_text(
            json.dumps(
                {
                    "slide_count": 1,
                    "slides": [
                        {
                            "slide_id": "slide_007",
                            "slide_index": 7,
                            "title": "RNN整体网络结构",
                            "text": "RNN整体网络结构 我 喜欢 打 篮球",
                            "relative_path": "slides_preview/rendered/slide_007.png",
                            "image_area": 1,
                            "image_frequency": 1,
                            "is_low_value": False,
                        }
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        raise subprocess.TimeoutExpired(command, timeout or 0, output="partial", stderr="")

    monkeypatch.setattr(video_to_notes, "run_command", fake_run_command)

    cleaned = run_codex_slides_cleanup(output_paths, raw_payload=raw_payload, timeout_seconds=12)

    report = json.loads(Path(output_paths["slides_cleanup_report_json"]).read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["generator"] == "codex"
    assert report["attempts"][0]["command_status"] == "timeout_with_valid_cleanup"
    assert cleaned["slides"][0]["title"] == "RNN整体网络结构"


def test_merge_segment_review_results_preserves_review_segment_order() -> None:
    review_segments = [
        {
            "segment_id": "segment_001",
            "start": 0.0,
            "end": 60.0,
            "label": "00:00:00-00:01:00",
            "segment_indexes": [1],
            "char_count": 10,
            "text": "segment 1",
            "ocr_hints": ["hint 1"],
        },
        {
            "segment_id": "segment_002",
            "start": 60.0,
            "end": 120.0,
            "label": "00:01:00-00:02:00",
            "segment_indexes": [2],
            "char_count": 10,
            "text": "segment 2",
            "ocr_hints": ["hint 2"],
        },
    ]
    results = {
        "segment_002": {
            "segment_id": "segment_002",
            "cleaned_text": "cleaned second",
            "payload": {
                "review_status": "done",
                "last_updated": "2026-03-13T04:00:00+08:00",
                "segment_reviews": [
                    {
                        "segment_id": "segment_002",
                        "start": 60.0,
                        "end": 120.0,
                        "summary": "second summary",
                        "issues": ["term 2"],
                        "status": "done",
                    }
                ],
                "corrections": [
                    {"raw": "Bort2", "cleaned": "BERT2", "reason": "model", "evidence": ["context 2"]}
                ],
            },
        },
        "segment_001": {
            "segment_id": "segment_001",
            "cleaned_text": "cleaned first",
            "payload": {
                "review_status": "done",
                "last_updated": "2026-03-13T04:00:00+08:00",
                "segment_reviews": [
                    {
                        "segment_id": "segment_001",
                        "start": 0.0,
                        "end": 60.0,
                        "summary": "first summary",
                        "issues": ["term 1"],
                        "status": "done",
                    }
                ],
                "corrections": [
                    {"raw": "Bort1", "cleaned": "BERT1", "reason": "model", "evidence": ["context 1"]}
                ],
            },
        },
    }

    merged_text, merged_payload = video_to_notes.merge_segment_review_results(
        review_segments=review_segments,
        segment_results=results,
        transcript_txt=Path("transcript.txt"),
        transcript_cleaned_txt=Path("transcript.cleaned.txt"),
    )

    assert merged_text == "cleaned first\ncleaned second"
    assert [item["segment_id"] for item in merged_payload["segment_reviews"]] == ["segment_001", "segment_002"]
    assert [item["raw"] for item in merged_payload["corrections"]] == ["Bort1", "Bort2"]


def test_review_artifacts_completed_detects_placeholder_payload(tmp_path: Path) -> None:
    raw = tmp_path / "transcript.txt"
    cleaned = tmp_path / "transcript.cleaned.txt"
    corrections = tmp_path / "transcript.corrections.json"
    review_segments = tmp_path / "review_segments.json"
    raw.write_text("raw", encoding="utf-8")
    cleaned.write_text("raw", encoding="utf-8")
    write_review_segments(review_segments)
    corrections.write_text(
        json.dumps(
            {
                "source_transcript": "transcript.txt",
                "cleaned_transcript": "transcript.cleaned.txt",
                "review_status": "pending",
                "last_updated": None,
                "segment_reviews": [
                    {
                        "segment_id": "segment_001",
                        "start": 0.0,
                        "end": 60.0,
                        "summary": "",
                        "issues": [],
                        "status": "pending",
                    }
                ],
                "corrections": [{"raw": "", "cleaned": "", "reason": "", "evidence": []}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    assert review_artifacts_completed(raw, cleaned, corrections, review_segments) is False


def test_explain_review_failure_reports_detailed_reasons(tmp_path: Path) -> None:
    raw = tmp_path / "transcript.txt"
    cleaned = tmp_path / "transcript.cleaned.txt"
    corrections = tmp_path / "transcript.corrections.json"
    review_segments = tmp_path / "review_segments.json"
    raw.write_text("Bort 研磨 下居 " * 80, encoding="utf-8")
    cleaned.write_text("Bort 研磨 下居 " * 80, encoding="utf-8")
    write_review_segments(review_segments, segment_count=2)
    corrections.write_text(
        json.dumps(
            {
                "source_transcript": "transcript.txt",
                "cleaned_transcript": "transcript.cleaned.txt",
                "review_status": "pending",
                "last_updated": None,
                "segment_reviews": [
                    {
                        "segment_id": "segment_001",
                        "start": 0.0,
                        "end": 60.0,
                        "summary": "",
                        "issues": [],
                        "status": "pending",
                    }
                ],
                "corrections": [{"raw": "", "cleaned": "", "reason": "", "evidence": []}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    reasons = explain_review_failure(raw, cleaned, corrections, review_segments)

    assert "transcript.cleaned.txt 与 transcript.txt 完全相同" in reasons
    assert "review_status 不是 done" in reasons
    assert "last_updated 为空" in reasons
    assert any("segment_reviews 未覆盖全部分段" in item for item in reasons)
    assert any("有效纠错条目不足" in item for item in reasons)


def test_review_artifacts_completed_rejects_identical_cleaned_text(tmp_path: Path) -> None:
    raw = tmp_path / "transcript.txt"
    cleaned = tmp_path / "transcript.cleaned.txt"
    corrections = tmp_path / "transcript.corrections.json"
    review_segments = tmp_path / "review_segments.json"
    raw_text = "Bort " * 80
    raw.write_text(raw_text, encoding="utf-8")
    cleaned.write_text(raw_text, encoding="utf-8")
    write_review_segments(review_segments)
    corrections.write_text(
        json.dumps(
            {
                "source_transcript": "transcript.txt",
                "cleaned_transcript": "transcript.cleaned.txt",
                "review_status": "done",
                "last_updated": "2026-03-13T04:00:00+08:00",
                "segment_reviews": [
                    {
                        "segment_id": "segment_001",
                        "start": 0.0,
                        "end": 60.0,
                        "summary": "讲 BERT",
                        "issues": ["Bort -> BERT"],
                        "status": "done",
                    }
                ],
                "corrections": [
                    {
                        "raw": "Bort",
                        "cleaned": "BERT",
                        "reason": "model name",
                        "evidence": ["context"],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    assert review_artifacts_completed(raw, cleaned, corrections, review_segments) is False


def test_review_artifacts_completed_requires_multiple_corrections_for_long_transcript(tmp_path: Path) -> None:
    raw = tmp_path / "transcript.txt"
    cleaned = tmp_path / "transcript.cleaned.txt"
    corrections = tmp_path / "transcript.corrections.json"
    review_segments = tmp_path / "review_segments.json"
    raw.write_text("Bort 研磨 下居 " * 80, encoding="utf-8")
    cleaned.write_text("BERT 掩码 下句 " * 80, encoding="utf-8")
    write_review_segments(review_segments, segment_count=3)
    corrections.write_text(
        json.dumps(
            {
                "source_transcript": "transcript.txt",
                "cleaned_transcript": "transcript.cleaned.txt",
                "review_status": "done",
                "last_updated": "2026-03-13T04:00:00+08:00",
                "segment_reviews": [
                    {
                        "segment_id": "segment_001",
                        "start": 0.0,
                        "end": 60.0,
                        "summary": "第一段",
                        "issues": ["Bort -> BERT"],
                        "status": "done",
                    },
                    {
                        "segment_id": "segment_002",
                        "start": 60.0,
                        "end": 120.0,
                        "summary": "第二段",
                        "issues": ["研磨 -> 掩码"],
                        "status": "done",
                    },
                    {
                        "segment_id": "segment_003",
                        "start": 120.0,
                        "end": 180.0,
                        "summary": "第三段",
                        "issues": ["下居 -> 下句"],
                        "status": "done",
                    },
                ],
                "corrections": [
                    {
                        "raw": "Bort",
                        "cleaned": "BERT",
                        "reason": "model name",
                        "evidence": ["context"],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    assert review_artifacts_completed(raw, cleaned, corrections, review_segments) is False


def test_run_codex_review_invokes_codex_and_requires_non_placeholder_result(
    tmp_path: Path, monkeypatch
) -> None:
    output_paths = plan_output_paths(Path("/tmp/4.1BERT-2训练.mp4"), tmp_path)
    work_dir = output_paths["work_dir"]
    video_to_notes.ensure_dirs(output_paths)
    Path(output_paths["transcript_txt"]).write_text("Bort", encoding="utf-8")
    Path(output_paths["transcript_cleaned_txt"]).write_text("Bort", encoding="utf-8")
    write_review_segments(Path(output_paths["review_segments_json"]))
    Path(output_paths["transcript_corrections_json"]).write_text(
        json.dumps(
            {
                "source_transcript": "transcript.txt",
                "cleaned_transcript": "transcript.cleaned.txt",
                "review_status": "pending",
                "last_updated": None,
                "segment_reviews": [],
                "corrections": [{"raw": "", "cleaned": "", "reason": "", "evidence": []}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    Path(output_paths["codex_review_prompt_md"]).write_text("review me", encoding="utf-8")

    calls: list[tuple[list[str], Path | None]] = []

    def fake_run_command(
        command: list[str], *, cwd: Path | None = None, timeout: int | None = None
    ) -> subprocess.CompletedProcess[str]:
        calls.append((command, cwd))
        assert cwd is not None
        write_segment_review_outputs(
            cwd,
            cleaned_text="BERT",
            issues=["Bort -> BERT"],
            corrections=[
                {
                    "raw": "Bort",
                    "cleaned": "BERT",
                    "reason": "model name",
                    "evidence": ["context"],
                }
            ],
        )
        return subprocess.CompletedProcess(command, 0, "ok", "")

    monkeypatch.setattr(video_to_notes, "run_command", fake_run_command)

    run_codex_review(output_paths)

    assert calls
    assert calls[0][0][:2] == ["codex", "exec"]
    assert "--full-auto" in calls[0][0]
    assert "-c" in calls[0][0]
    config_index = calls[0][0].index("-c")
    assert calls[0][0][config_index + 1] == 'model_reasoning_effort="low"'
    assert calls[0][1] != Path(output_paths["work_dir"])
    assert Path(tempfile.gettempdir()) in Path(calls[0][1]).parents
    report = json.loads(Path(output_paths["review_report_json"]).read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["attempt_count"] == 1
    assert report["planned_segment_count"] == 1
    assert report["reviewed_segment_count"] == 1
    assert report["valid_correction_count"] == 1
    assert report["attempts"][0]["passed"] is True
    assert Path(output_paths["transcript_cleaned_txt"]).read_text(encoding="utf-8") == "BERT"


def test_run_codex_review_retries_once_with_failure_feedback(tmp_path: Path, monkeypatch) -> None:
    output_paths = plan_output_paths(Path("/tmp/4.1BERT-2训练.mp4"), tmp_path)
    work_dir = output_paths["work_dir"]
    video_to_notes.ensure_dirs(output_paths)
    Path(output_paths["transcript_txt"]).write_text("Bort 研磨 下居 " * 80, encoding="utf-8")
    Path(output_paths["transcript_cleaned_txt"]).write_text("Bort 研磨 下居 " * 80, encoding="utf-8")
    write_review_segments(Path(output_paths["review_segments_json"]), segment_count=1)
    Path(output_paths["transcript_corrections_json"]).write_text(
        json.dumps(
            {
                "source_transcript": "transcript.txt",
                "cleaned_transcript": "transcript.cleaned.txt",
                "review_status": "pending",
                "last_updated": None,
                "segment_reviews": [],
                "corrections": [{"raw": "", "cleaned": "", "reason": "", "evidence": []}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    Path(output_paths["codex_review_prompt_md"]).write_text("review me", encoding="utf-8")

    calls: list[list[str]] = []

    def fake_run_command(
        command: list[str], *, cwd: Path | None = None, timeout: int | None = None
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if len(calls) == 1:
            return subprocess.CompletedProcess(command, 0, "first pass failed quality", "")

        assert cwd is not None
        write_segment_review_outputs(
            cwd,
            cleaned_text="BERT 掩码 下句 " * 80,
            issues=["Bort -> BERT", "研磨 -> 掩码", "下居 -> 下句"],
            corrections=[
                {"raw": "Bort", "cleaned": "BERT", "reason": "model name", "evidence": ["context"]},
                {"raw": "研磨", "cleaned": "掩码", "reason": "term", "evidence": ["context"]},
                {"raw": "下居", "cleaned": "下句", "reason": "term", "evidence": ["context"]},
            ],
        )
        return subprocess.CompletedProcess(command, 0, "second pass ok", "")

    monkeypatch.setattr(video_to_notes, "run_command", fake_run_command)

    run_codex_review(output_paths)

    assert len(calls) == 2
    assert "上一次这个 segment 的审阅结果未通过质量校验" in calls[1][-1]
    report = json.loads(Path(output_paths["review_report_json"]).read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["attempt_count"] == 2
    assert report["planned_segment_count"] == 1
    assert report["reviewed_segment_count"] == 1
    assert report["attempts"][0]["passed"] is False
    assert report["attempts"][1]["passed"] is True
    assert any("segment_reviews 未覆盖全部分段" in item for item in report["attempts"][0]["failure_reasons"])
    assert any("有效纠错条目不足" in item for item in report["attempts"][0]["failure_reasons"])


def test_run_codex_review_retries_with_json_only_prompt_when_cleaned_transcript_already_passes(
    tmp_path: Path, monkeypatch
) -> None:
    output_paths = plan_output_paths(Path("/tmp/4.1BERT-2训练.mp4"), tmp_path)
    work_dir = output_paths["work_dir"]
    video_to_notes.ensure_dirs(output_paths)
    Path(output_paths["transcript_txt"]).write_text("Bort 研磨 下居 " * 80, encoding="utf-8")
    Path(output_paths["transcript_cleaned_txt"]).write_text("Bort 研磨 下居 " * 80, encoding="utf-8")
    write_review_segments(Path(output_paths["review_segments_json"]), segment_count=1)
    Path(output_paths["transcript_corrections_json"]).write_text(
        json.dumps(
            {
                "source_transcript": "transcript.txt",
                "cleaned_transcript": "transcript.cleaned.txt",
                "review_status": "pending",
                "last_updated": None,
                "segment_reviews": [],
                "corrections": [{"raw": "", "cleaned": "", "reason": "", "evidence": []}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    Path(output_paths["codex_review_prompt_md"]).write_text("review me", encoding="utf-8")

    calls: list[list[str]] = []

    def fake_run_command(
        command: list[str], *, cwd: Path | None = None, timeout: int | None = None
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        assert cwd is not None
        if len(calls) == 1:
            Path(cwd / "segment.cleaned.txt").write_text("BERT 掩码 下句 " * 80, encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, "cleaned transcript only", "")

        write_segment_review_outputs(
            cwd,
            cleaned_text="BERT 掩码 下句 " * 80,
            issues=["Bort -> BERT", "研磨 -> 掩码", "下居 -> 下句"],
            corrections=[
                {"raw": "Bort", "cleaned": "BERT", "reason": "model name", "evidence": ["context"]},
                {"raw": "研磨", "cleaned": "掩码", "reason": "term", "evidence": ["context"]},
                {"raw": "下居", "cleaned": "下句", "reason": "term", "evidence": ["context"]},
            ],
        )
        return subprocess.CompletedProcess(command, 0, "json completed", "")

    monkeypatch.setattr(video_to_notes, "run_command", fake_run_command)

    run_codex_review(output_paths)

    assert len(calls) == 2
    retry_prompt = calls[1][-1]
    assert "只需要补全 segment.corrections.json" in retry_prompt
    assert "不要重写 segment.cleaned.txt" in retry_prompt


def test_run_codex_review_raises_after_retry_failure(tmp_path: Path, monkeypatch) -> None:
    output_paths = plan_output_paths(Path("/tmp/4.1BERT-2训练.mp4"), tmp_path)
    work_dir = output_paths["work_dir"]
    video_to_notes.ensure_dirs(output_paths)
    Path(output_paths["transcript_txt"]).write_text("Bort 研磨 下居 " * 80, encoding="utf-8")
    Path(output_paths["transcript_cleaned_txt"]).write_text("Bort 研磨 下居 " * 80, encoding="utf-8")
    write_review_segments(Path(output_paths["review_segments_json"]), segment_count=1)
    Path(output_paths["transcript_corrections_json"]).write_text(
        json.dumps(
            {
                "source_transcript": "transcript.txt",
                "cleaned_transcript": "transcript.cleaned.txt",
                "review_status": "pending",
                "last_updated": None,
                "segment_reviews": [],
                "corrections": [{"raw": "", "cleaned": "", "reason": "", "evidence": []}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    Path(output_paths["codex_review_prompt_md"]).write_text("review me", encoding="utf-8")

    calls: list[list[str]] = []

    def fake_run_command(
        command: list[str], *, cwd: Path | None = None, timeout: int | None = None
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, f"attempt {len(calls)}", "")

    monkeypatch.setattr(video_to_notes, "run_command", fake_run_command)

    try:
        run_codex_review(output_paths)
    except RuntimeError as exc:
        assert "quality gate failed after retry" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")

    assert len(calls) == 2
    report = json.loads(Path(output_paths["review_report_json"]).read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["attempt_count"] == 2
    assert report["planned_segment_count"] == 1
    assert report["reviewed_segment_count"] == 0
    assert report["missing_segment_ids"] == ["segment_001"]
    assert report["attempts"][0]["passed"] is False
    assert report["attempts"][1]["passed"] is False


def test_run_codex_review_reports_timeout_and_raises_after_retry(tmp_path: Path, monkeypatch) -> None:
    output_paths = plan_output_paths(Path("/tmp/4.1BERT-2训练.mp4"), tmp_path)
    work_dir = output_paths["work_dir"]
    video_to_notes.ensure_dirs(output_paths)
    Path(output_paths["transcript_txt"]).write_text("Bort 研磨 下居 " * 80, encoding="utf-8")
    Path(output_paths["transcript_cleaned_txt"]).write_text("Bort 研磨 下居 " * 80, encoding="utf-8")
    write_review_segments(Path(output_paths["review_segments_json"]), segment_count=1)
    Path(output_paths["transcript_corrections_json"]).write_text(
        json.dumps(
            {
                "source_transcript": "transcript.txt",
                "cleaned_transcript": "transcript.cleaned.txt",
                "review_status": "pending",
                "last_updated": None,
                "segment_reviews": [],
                "corrections": [{"raw": "", "cleaned": "", "reason": "", "evidence": []}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    Path(output_paths["codex_review_prompt_md"]).write_text("review me", encoding="utf-8")

    def fake_run_command(
        command: list[str], *, cwd: Path | None = None, timeout: int | None = None
    ) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(command, timeout or 1)

    monkeypatch.setattr(video_to_notes, "run_command", fake_run_command)

    try:
        run_codex_review(output_paths, timeout_seconds=12)
    except RuntimeError as exc:
        assert "quality gate failed after retry" in str(exc)
        assert "codex exec 超时" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")

    report = json.loads(Path(output_paths["review_report_json"]).read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["attempt_count"] == 2
    assert "codex exec 超时（>12 秒）" in report["attempts"][0]["failure_reasons"]
    assert "codex exec 超时（>12 秒）" in report["attempts"][1]["failure_reasons"]


def test_run_codex_review_accepts_completed_artifacts_even_if_process_times_out(
    tmp_path: Path, monkeypatch
) -> None:
    output_paths = plan_output_paths(Path("/tmp/4.1BERT-2训练.mp4"), tmp_path)
    work_dir = output_paths["work_dir"]
    video_to_notes.ensure_dirs(output_paths)
    Path(output_paths["transcript_txt"]).write_text("Bort 研磨 下居 " * 80, encoding="utf-8")
    Path(output_paths["transcript_cleaned_txt"]).write_text("Bort 研磨 下居 " * 80, encoding="utf-8")
    write_review_segments(Path(output_paths["review_segments_json"]), segment_count=1)
    Path(output_paths["transcript_corrections_json"]).write_text(
        json.dumps(
            {
                "source_transcript": "transcript.txt",
                "cleaned_transcript": "transcript.cleaned.txt",
                "review_status": "pending",
                "last_updated": None,
                "segment_reviews": [],
                "corrections": [{"raw": "", "cleaned": "", "reason": "", "evidence": []}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    Path(output_paths["codex_review_prompt_md"]).write_text("review me", encoding="utf-8")

    def fake_run_command(
        command: list[str], *, cwd: Path | None = None, timeout: int | None = None
    ) -> subprocess.CompletedProcess[str]:
        assert cwd is not None
        write_segment_review_outputs(
            cwd,
            cleaned_text="BERT 掩码 下句 " * 80,
            issues=["Bort -> BERT", "研磨 -> 掩码", "下居 -> 下句"],
            corrections=[
                {"raw": "Bort", "cleaned": "BERT", "reason": "model name", "evidence": ["context"]},
                {"raw": "研磨", "cleaned": "掩码", "reason": "term", "evidence": ["context"]},
                {"raw": "下居", "cleaned": "下句", "reason": "term", "evidence": ["context"]},
            ],
        )
        raise subprocess.TimeoutExpired(command, timeout or 1, output="partial")

    monkeypatch.setattr(video_to_notes, "run_command", fake_run_command)

    run_codex_review(output_paths, timeout_seconds=12)

    report = json.loads(Path(output_paths["review_report_json"]).read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["attempt_count"] == 1
    assert report["attempts"][0]["passed"] is True
    assert report["attempts"][0]["timed_out"] is True


def test_run_codex_review_accepts_completed_artifacts_even_if_process_exits_nonzero(
    tmp_path: Path, monkeypatch
) -> None:
    output_paths = plan_output_paths(Path("/tmp/4.1BERT-2训练.mp4"), tmp_path)
    work_dir = output_paths["work_dir"]
    video_to_notes.ensure_dirs(output_paths)
    Path(output_paths["transcript_txt"]).write_text("Bort 研磨 下居 " * 80, encoding="utf-8")
    Path(output_paths["transcript_cleaned_txt"]).write_text("Bort 研磨 下居 " * 80, encoding="utf-8")
    write_review_segments(Path(output_paths["review_segments_json"]), segment_count=1)
    Path(output_paths["transcript_corrections_json"]).write_text(
        json.dumps(
            {
                "source_transcript": "transcript.txt",
                "cleaned_transcript": "transcript.cleaned.txt",
                "review_status": "pending",
                "last_updated": None,
                "segment_reviews": [],
                "corrections": [{"raw": "", "cleaned": "", "reason": "", "evidence": []}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    Path(output_paths["codex_review_prompt_md"]).write_text("review me", encoding="utf-8")

    def fake_run_command(
        command: list[str], *, cwd: Path | None = None, timeout: int | None = None
    ) -> subprocess.CompletedProcess[str]:
        assert cwd is not None
        write_segment_review_outputs(
            cwd,
            cleaned_text="BERT 掩码 下句 " * 80,
            issues=["Bort -> BERT", "研磨 -> 掩码", "下居 -> 下句"],
            corrections=[
                {"raw": "Bort", "cleaned": "BERT", "reason": "model name", "evidence": ["context"]},
                {"raw": "研磨", "cleaned": "掩码", "reason": "term", "evidence": ["context"]},
                {"raw": "下居", "cleaned": "下句", "reason": "term", "evidence": ["context"]},
            ],
        )
        raise subprocess.CalledProcessError(1, command, output="partial stdout", stderr="partial stderr")

    monkeypatch.setattr(video_to_notes, "run_command", fake_run_command)

    run_codex_review(output_paths)

    report = json.loads(Path(output_paths["review_report_json"]).read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["attempt_count"] == 1
    assert report["attempts"][0]["passed"] is True
    assert report["attempts"][0]["timed_out"] is False


def test_run_codex_review_launches_one_segment_job_per_segment_and_merges_outputs(
    tmp_path: Path, monkeypatch
) -> None:
    output_paths = plan_output_paths(Path("/tmp/4.1BERT-2训练.mp4"), tmp_path)
    work_dir = Path(output_paths["work_dir"])
    video_to_notes.ensure_dirs(output_paths)
    Path(output_paths["transcript_txt"]).write_text("segment 1 raw\nsegment 2 raw", encoding="utf-8")
    Path(output_paths["transcript_cleaned_txt"]).write_text("segment 1 raw\nsegment 2 raw", encoding="utf-8")
    write_review_segments(Path(output_paths["review_segments_json"]), segment_count=2)
    Path(output_paths["transcript_corrections_json"]).write_text(
        json.dumps(
            {
                "source_transcript": "transcript.txt",
                "cleaned_transcript": "transcript.cleaned.txt",
                "review_status": "pending",
                "last_updated": None,
                "segment_reviews": [],
                "corrections": [{"raw": "", "cleaned": "", "reason": "", "evidence": []}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    Path(output_paths["codex_review_prompt_md"]).write_text("review me", encoding="utf-8")

    calls: list[str] = []

    def fake_run_command(
        command: list[str], *, cwd: Path | None = None, timeout: int | None = None
    ) -> subprocess.CompletedProcess[str]:
        assert cwd is not None
        segment_input = json.loads((cwd / "segment_input.json").read_text(encoding="utf-8"))
        segment_id = segment_input["segment"]["segment_id"]
        calls.append(segment_id)
        if segment_id == "segment_001":
            time.sleep(0.05)
        write_segment_review_outputs(
            cwd,
            cleaned_text=f"{segment_id} cleaned",
            issues=[f"{segment_id} fix"],
            corrections=[
                {
                    "raw": segment_id,
                    "cleaned": f"{segment_id} cleaned",
                    "reason": "context",
                    "evidence": ["context"],
                }
            ],
        )
        return subprocess.CompletedProcess(command, 0, f"{segment_id} ok", "")

    monkeypatch.setattr(video_to_notes, "run_command", fake_run_command)

    run_codex_review(output_paths, parallelism=2)

    assert sorted(calls) == ["segment_001", "segment_002"]
    assert Path(output_paths["transcript_cleaned_txt"]).read_text(encoding="utf-8") == (
        "segment_001 cleaned\nsegment_002 cleaned"
    )
    merged = json.loads(Path(output_paths["transcript_corrections_json"]).read_text(encoding="utf-8"))
    assert [item["segment_id"] for item in merged["segment_reviews"]] == ["segment_001", "segment_002"]
    assert [item["raw"] for item in merged["corrections"]] == ["segment_001", "segment_002"]
    report = json.loads(Path(output_paths["review_report_json"]).read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["planned_segment_count"] == 2
    assert report["reviewed_segment_count"] == 2
    assert report["attempt_count"] == 2


def test_run_codex_review_accepts_globally_valid_merge_even_if_individual_segment_is_marked_failed(
    tmp_path: Path, monkeypatch
) -> None:
    output_paths = plan_output_paths(Path("/tmp/4.1BERT-2训练.mp4"), tmp_path)
    video_to_notes.ensure_dirs(output_paths)
    raw_text = "Bort 研磨 下居 " * 80
    Path(output_paths["transcript_txt"]).write_text(raw_text, encoding="utf-8")
    Path(output_paths["transcript_cleaned_txt"]).write_text(raw_text, encoding="utf-8")
    write_review_segments(Path(output_paths["review_segments_json"]), segment_count=2)
    Path(output_paths["transcript_corrections_json"]).write_text(
        json.dumps(
            {
                "source_transcript": "transcript.txt",
                "cleaned_transcript": "transcript.cleaned.txt",
                "review_status": "pending",
                "last_updated": None,
                "segment_reviews": [],
                "corrections": [{"raw": "", "cleaned": "", "reason": "", "evidence": []}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    Path(output_paths["codex_review_prompt_md"]).write_text("review me", encoding="utf-8")

    review_segments = video_to_notes.load_review_segments(Path(output_paths["review_segments_json"]))

    def fake_run_segment_codex_review(**kwargs):
        segment = kwargs["segment"]
        segment_id = segment["segment_id"]
        payload = video_to_notes.correction_template(
            Path("segment_transcript.txt"),
            Path("segment.cleaned.txt"),
            [segment],
        )
        payload["review_status"] = "done"
        payload["last_updated"] = "2026-03-15T08:00:00+08:00"
        payload["segment_reviews"] = [
            {
                "segment_id": segment_id,
                "start": segment["start"],
                "end": segment["end"],
                "summary": f"{segment_id} summary",
                "issues": [f"{segment_id} issue"],
                "status": "done",
            }
        ]
        payload["corrections"] = [
            {
                "raw": f"{segment_id}-raw-1",
                "cleaned": f"{segment_id}-clean-1",
                "reason": "context",
                "evidence": ["context"],
            },
            {
                "raw": f"{segment_id}-raw-2",
                "cleaned": f"{segment_id}-clean-2",
                "reason": "context",
                "evidence": ["context"],
            },
            {
                "raw": f"{segment_id}-raw-3",
                "cleaned": f"{segment_id}-clean-3",
                "reason": "context",
                "evidence": ["context"],
            },
        ]
        return {
            "segment_id": segment_id,
            "passed": segment_id != "segment_001",
            "attempt_count": 1,
            "attempts": [
                {
                    "segment_id": segment_id,
                    "attempt": 1,
                    "prompt_type": "initial",
                    "passed": segment_id != "segment_001",
                    "timed_out": segment_id == "segment_001",
                    "failure_reasons": ["codex exec 超时（>60 秒）"] if segment_id == "segment_001" else [],
                    "codex_output_excerpt": "ok",
                }
            ],
            "failure_reasons": ["codex exec 超时（>60 秒）"] if segment_id == "segment_001" else [],
            "last_detail": "ok",
            "cleaned_text": f"{segment_id} cleaned",
            "payload": payload,
        }

    monkeypatch.setattr(video_to_notes, "run_segment_codex_review", fake_run_segment_codex_review)

    run_codex_review(output_paths, parallelism=2)

    report = json.loads(Path(output_paths["review_report_json"]).read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["planned_segment_count"] == len(review_segments)
    assert report["reviewed_segment_count"] == len(review_segments)
    assert report["missing_segment_ids"] == []
    merged = json.loads(Path(output_paths["transcript_corrections_json"]).read_text(encoding="utf-8"))
    assert len(merged["segment_reviews"]) == 2
    assert len(merged["corrections"]) == 6


def test_run_codex_review_retries_only_failed_segment_with_json_only_prompt(
    tmp_path: Path, monkeypatch
) -> None:
    output_paths = plan_output_paths(Path("/tmp/4.1BERT-2训练.mp4"), tmp_path)
    work_dir = Path(output_paths["work_dir"])
    video_to_notes.ensure_dirs(output_paths)
    raw_text = "Bort 研磨 " * 60
    Path(output_paths["transcript_txt"]).write_text(raw_text, encoding="utf-8")
    Path(output_paths["transcript_cleaned_txt"]).write_text(raw_text, encoding="utf-8")
    write_review_segments(Path(output_paths["review_segments_json"]), segment_count=2)
    Path(output_paths["transcript_corrections_json"]).write_text(
        json.dumps(
            {
                "source_transcript": "transcript.txt",
                "cleaned_transcript": "transcript.cleaned.txt",
                "review_status": "pending",
                "last_updated": None,
                "segment_reviews": [],
                "corrections": [{"raw": "", "cleaned": "", "reason": "", "evidence": []}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    Path(output_paths["codex_review_prompt_md"]).write_text("review me", encoding="utf-8")

    call_log: list[tuple[str, str]] = []
    attempts_by_segment: dict[str, int] = {"segment_001": 0, "segment_002": 0}

    def fake_run_command(
        command: list[str], *, cwd: Path | None = None, timeout: int | None = None
    ) -> subprocess.CompletedProcess[str]:
        assert cwd is not None
        segment_input = json.loads((cwd / "segment_input.json").read_text(encoding="utf-8"))
        segment_id = segment_input["segment"]["segment_id"]
        prompt = command[-1]
        attempts_by_segment[segment_id] += 1
        call_log.append((segment_id, prompt))

        if segment_id == "segment_001":
            write_segment_review_outputs(
                cwd,
                cleaned_text="segment_001 cleaned",
                issues=["segment_001 fix"],
                corrections=[
                    {
                        "raw": "Bort",
                        "cleaned": "BERT",
                        "reason": "model",
                        "evidence": ["context"],
                    }
                ],
            )
            return subprocess.CompletedProcess(command, 0, "segment_001 ok", "")

        if attempts_by_segment[segment_id] == 1:
            (cwd / "segment.cleaned.txt").write_text("segment_002 cleaned", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, "segment_002 cleaned only", "")

        write_segment_review_outputs(
            cwd,
            cleaned_text="segment_002 cleaned",
            issues=["segment_002 fix"],
            corrections=[
                {
                    "raw": "研磨",
                    "cleaned": "掩码",
                    "reason": "term",
                    "evidence": ["context"],
                },
                {
                    "raw": "Bort second",
                    "cleaned": "BERT second",
                    "reason": "model",
                    "evidence": ["context"],
                },
            ],
        )
        return subprocess.CompletedProcess(command, 0, "segment_002 json ok", "")

    monkeypatch.setattr(video_to_notes, "run_command", fake_run_command)

    run_codex_review(output_paths, parallelism=2)

    assert attempts_by_segment == {"segment_001": 1, "segment_002": 2}
    segment_002_prompts = [prompt for segment_id, prompt in call_log if segment_id == "segment_002"]
    assert len(segment_002_prompts) == 2
    assert "只需要补全 segment.corrections.json" in segment_002_prompts[-1]
    report = json.loads(Path(output_paths["review_report_json"]).read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["attempt_count"] == 3
    assert report["reviewed_segment_count"] == 2
    assert report["segment_results"]["segment_001"]["attempt_count"] == 1
    assert report["segment_results"]["segment_002"]["attempt_count"] == 2

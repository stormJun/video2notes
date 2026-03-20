from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
from typing import Any

from video_to_notes import (
    build_note_markdown,
    build_note_blocks,
    build_note_generation_prompt,
    build_note_outline,
    build_slides_cleanup_prompt,
    render_ppt_alignment_debug_markdown,
    build_visual_alignment,
    build_visual_units,
    ensure_dirs,
    ensure_review_artifacts,
    explain_note_generation_failure,
    extract_audio,
    extract_frames_at_timestamps,
    ffprobe_duration,
    load_transcript_segments,
    move_whisper_outputs,
    merge_visual_candidates,
    ocr_frames,
    parse_args,
    plan_visual_supplemental_timestamps,
    plan_output_paths,
    preferred_transcript_path,
    read_excerpt,
    require_binary,
    run_codex_slides_cleanup,
    run_codex_note_generation,
    run_codex_review,
    scan_visual_candidate_timestamps,
    transcribe_audio,
    write_slides_cleanup_report,
    write_note_generation_report,
    write_metadata,
)
from video_to_notes_artifacts import resolve_stage_plan, sync_pipeline_artifact_view
from video_to_notes_slides import (
    assess_slides_payload_for_video,
    determine_visual_source_mode,
    prepare_slides_payload,
    resolve_slides_path,
    sanitize_slides_payload,
    slides_payload_has_noise,
)


def _force_flags(args: Any) -> dict[str, bool]:
    return {
        "audio": bool(getattr(args, "force_audio", False)),
        "transcribe": bool(getattr(args, "force_transcribe", False)),
        "frames": bool(getattr(args, "force_frames", False)),
        "ocr": bool(getattr(args, "force_ocr", False)),
        "review_artifacts": bool(getattr(args, "force_review_artifacts", False)),
        "codex_review": bool(getattr(args, "force_codex_review", False)),
        "note": bool(getattr(args, "force_note", False)),
    }


def _load_existing_frames(output_paths: dict[str, Path | str]) -> list[dict[str, Any]]:
    ocr_json = Path(output_paths["visual_candidates_ocr_json"])
    if not ocr_json.exists():
        return []
    try:
        payload = json.loads(ocr_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return payload if isinstance(payload, list) else []


def _load_frames_from_visual_candidates(output_paths: dict[str, Path | str]) -> list[dict[str, Any]]:
    visual_candidates_dir = Path(output_paths["visual_candidates_dir"])
    work_dir = Path(output_paths["work_dir"])
    frames: list[dict[str, Any]] = []
    for frame_path in sorted(visual_candidates_dir.glob("frame_*.jpg")):
        frames.append(
            {
                "timestamp": 0.0,
                "path": str(frame_path),
                "relative_path": frame_path.relative_to(work_dir).as_posix(),
            }
        )
    return frames


def _load_json_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_cleaned_segments(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _remove_obsolete_pipeline_outputs(output_paths: dict[str, Path | str]) -> None:
    pipeline_dir = Path(output_paths["pipeline_dir"])
    obsolete_paths = [
        pipeline_dir / "03_visual_source",
        pipeline_dir / "04_alignment",
        pipeline_dir / "05_note" / "note_outline.json",
        pipeline_dir / "05_note" / "note_blocks.json",
    ]
    for obsolete_path in obsolete_paths:
        if obsolete_path.is_dir():
            shutil.rmtree(obsolete_path, ignore_errors=True)
        else:
            obsolete_path.unlink(missing_ok=True)


def _print_stage_plan(stage_plan: dict[str, dict[str, Any]]) -> None:
    for stage, decision in stage_plan.items():
        action = "RUN" if decision["run"] else "SKIP"
        print(f"{action} {stage}: {decision['reason']}")


def _prepare_pipeline_context(args: Any) -> dict[str, Any]:
    input_video = Path(args.input).resolve()
    output_root = Path(args.output_root).resolve()
    if not input_video.exists():
        raise SystemExit(f"Input video not found: {input_video}")

    require_binary("ffmpeg")
    require_binary("ffprobe")
    require_binary("tesseract")

    output_paths = plan_output_paths(input_video, output_root)
    ensure_dirs(output_paths)
    _remove_obsolete_pipeline_outputs(output_paths)
    sync_pipeline_artifact_view(output_paths)
    explicit_slides_path = getattr(args, "slides", None)
    slides_path = resolve_slides_path(input_video, explicit_slides_path)
    duration_seconds = ffprobe_duration(input_video)
    stage_plan = resolve_stage_plan(
        output_paths=output_paths,
        duration_seconds=duration_seconds,
        frame_interval=args.frame_interval,
        max_frames=args.max_frames,
        force=_force_flags(args),
    )
    _print_stage_plan(stage_plan)
    return {
        "args": args,
        "input_video": input_video,
        "output_paths": output_paths,
        "slides_path": slides_path,
        "duration_seconds": duration_seconds,
        "stage_plan": stage_plan,
    }


def _clear_frame_outputs(output_paths: dict[str, Path | str]) -> None:
    directory = Path(output_paths["visual_candidates_dir"])
    if directory.exists():
        for existing in directory.glob("frame_*.jpg"):
            existing.unlink()


def _refresh_visual_units(
    *,
    input_video: Path,
    output_paths: dict[str, Path | str],
    duration_seconds: float,
    stage_plan: dict[str, dict[str, Any]],
    frames: list[dict[str, Any]],
) -> None:
    visual_units_path = Path(output_paths["visual_units_json"])
    current_frames = list(frames)
    if stage_plan["frames"]["run"] or stage_plan["ocr"]["run"]:
        initial_visual_units = build_visual_units(current_frames)
        weak_candidate_timestamps = plan_visual_supplemental_timestamps(
            duration_seconds=duration_seconds,
            candidate_frames=[
                {
                    "timestamp": unit.get("representative_timestamp", 0.0),
                    "ocr_text": unit.get("ocr_text", ""),
                    "is_low_value": unit.get("is_low_value", False),
                }
                for unit in initial_visual_units.get("units", [])
            ],
        )
        if weak_candidate_timestamps:
            supplemental_candidates = extract_frames_at_timestamps(
                input_video,
                Path(output_paths["visual_candidates_dir"]),
                timestamps=weak_candidate_timestamps,
            )
            supplemental_candidates = ocr_frames(
                supplemental_candidates,
                Path(output_paths["visual_candidates_ocr_json"]),
            )
            current_frames = merge_visual_candidates(current_frames, supplemental_candidates)
            Path(output_paths["visual_candidates_ocr_json"]).write_text(
                json.dumps(current_frames, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        visual_units_path.write_text(
            json.dumps(build_visual_units(current_frames), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    elif not visual_units_path.exists():
        source_frames = current_frames or _load_existing_frames(output_paths)
        visual_units_path.write_text(
            json.dumps(build_visual_units(source_frames), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def run_audio_branch(context: dict[str, Any]) -> dict[str, Any]:
    args = context["args"]
    input_video = context["input_video"]
    output_paths = context["output_paths"]
    stage_plan = context["stage_plan"]

    if stage_plan["audio"]["run"]:
        extract_audio(input_video, Path(output_paths["audio_path"]))

    if stage_plan["transcribe"]["run"]:
        transcribe_audio(
            Path(output_paths["audio_path"]),
            Path(output_paths["work_dir"]),
            model=args.whisper_model,
            language=args.language,
        )
        move_whisper_outputs(output_paths)

    return {
        "transcript_segments": load_transcript_segments(Path(output_paths["transcript_json"])),
    }


def run_visual_branch(context: dict[str, Any]) -> dict[str, Any]:
    args = context["args"]
    input_video = context["input_video"]
    output_paths = context["output_paths"]
    duration_seconds = context["duration_seconds"]
    stage_plan = context["stage_plan"]

    frames = _load_existing_frames(output_paths)
    if stage_plan["frames"]["run"]:
        _clear_frame_outputs(output_paths)
        visual_candidate_timestamps = scan_visual_candidate_timestamps(
            input_video,
            duration_seconds=duration_seconds,
            max_candidates=max(args.max_frames * 4, 18),
        )
        frames = extract_frames_at_timestamps(
            input_video,
            Path(output_paths["visual_candidates_dir"]),
            timestamps=visual_candidate_timestamps,
        )

    if stage_plan["ocr"]["run"]:
        if not frames:
            frames = _load_frames_from_visual_candidates(output_paths)
        frames = ocr_frames(frames, Path(output_paths["visual_candidates_ocr_json"]))

    _refresh_visual_units(
        input_video=input_video,
        output_paths=output_paths,
        duration_seconds=duration_seconds,
        stage_plan=stage_plan,
        frames=frames,
    )

    return {"frames": frames}


def run_media_pipeline(context: dict[str, Any]) -> dict[str, Any]:
    stage_plan = context["stage_plan"]
    audio_work = stage_plan["audio"]["run"] or stage_plan["transcribe"]["run"]
    visual_work = stage_plan["frames"]["run"] or stage_plan["ocr"]["run"] or not Path(context["output_paths"]["visual_units_json"]).exists()

    audio_result: dict[str, Any] = {}
    visual_result: dict[str, Any] = {}
    if audio_work and visual_work:
        with ThreadPoolExecutor(max_workers=2) as executor:
            audio_future = executor.submit(run_audio_branch, context)
            visual_future = executor.submit(run_visual_branch, context)
            audio_result = audio_future.result()
            visual_result = visual_future.result()
    else:
        audio_result = run_audio_branch(context)
        visual_result = run_visual_branch(context)

    result = {
        "transcript_segments": audio_result.get("transcript_segments", []),
        "frames": visual_result.get("frames", []),
    }
    sync_pipeline_artifact_view(context["output_paths"])
    return result


def run_review_pipeline(context: dict[str, Any], media_result: dict[str, Any]) -> dict[str, Any]:
    args = context["args"]
    input_video = context["input_video"]
    output_paths = context["output_paths"]
    stage_plan = context["stage_plan"]
    frames = media_result["frames"]

    if stage_plan["review_artifacts"]["run"]:
        ensure_review_artifacts(
            source_video=input_video,
            transcript_txt=Path(output_paths["transcript_txt"]),
            transcript_cleaned_txt=Path(output_paths["transcript_cleaned_txt"]),
            transcript_corrections_json=Path(output_paths["transcript_corrections_json"]),
            codex_review_prompt_md=Path(output_paths["codex_review_prompt_md"]),
            work_dir_agents_md=Path(output_paths["work_dir_agents_md"]),
            transcript_json=Path(output_paths["transcript_json"]),
            review_segments_json=Path(output_paths["review_segments_json"]),
            frames=frames,
        )

    if not args.skip_codex_review and stage_plan["codex_review"]["run"]:
        require_binary("codex")
        run_codex_review(
            output_paths,
            timeout_seconds=args.codex_timeout_seconds,
            parallelism=args.codex_review_parallelism,
        )

    result = {
        "review_segments_payload": _load_json_payload(Path(output_paths["review_segments_json"])),
        "corrections_payload": _load_json_payload(Path(output_paths["transcript_corrections_json"])),
    }
    sync_pipeline_artifact_view(output_paths)
    return result


def run_alignment_pipeline(
    context: dict[str, Any],
    media_result: dict[str, Any],
    review_result: dict[str, Any],
) -> dict[str, Any]:
    args = context["args"]
    input_video = context["input_video"]
    output_paths = context["output_paths"]
    stage_plan = context["stage_plan"]
    slides_path = context["slides_path"]
    frames = media_result["frames"]
    review_segments_payload = review_result["review_segments_payload"]
    corrections_payload = review_result["corrections_payload"]
    slides_payload: dict[str, Any] = {}
    effective_visual_source_mode = "video-first"

    slides_index_path = Path(output_paths["slides_index_json"])
    slides_index_raw_path = Path(output_paths["slides_index_raw_json"])
    slides_preview_dir = Path(output_paths["slides_preview_dir"])
    if slides_path:
        raw_slides_payload = prepare_slides_payload(slides_path, slides_preview_dir)
        slides_index_raw_path.write_text(json.dumps(raw_slides_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        slides_payload = sanitize_slides_payload(raw_slides_payload)
        slides_index_path.write_text(json.dumps(slides_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        if slides_payload_has_noise(raw_slides_payload):
            Path(output_paths["slides_cleanup_prompt_md"]).write_text(
                build_slides_cleanup_prompt(output_paths),
                encoding="utf-8",
            )
            if not bool(getattr(args, "skip_codex_note", False)):
                require_binary("codex")
                slides_payload = run_codex_slides_cleanup(
                    output_paths,
                    raw_payload=raw_slides_payload,
                    timeout_seconds=getattr(args, "codex_timeout_seconds", 300),
                )
            else:
                write_slides_cleanup_report(
                    output_paths,
                    generator="rule_based",
                    attempts=[
                        {
                            "attempt": 1,
                            "prompt_type": "rule_based_sanitize",
                            "command_status": "completed",
                            "failure_reasons": [],
                            "detail_excerpt": "",
                            "quality_gate_passed": True,
                        }
                    ],
                    status="passed",
                    final_failure_reasons=[],
                    noise_detected_before=True,
                    noise_detected_after=slides_payload_has_noise(slides_payload),
                )
        else:
            write_slides_cleanup_report(
                output_paths,
                generator="rule_based",
                attempts=[
                    {
                        "attempt": 1,
                        "prompt_type": "raw_payload_already_clean",
                        "command_status": "completed",
                        "failure_reasons": [],
                        "detail_excerpt": "",
                        "quality_gate_passed": True,
                    }
                ],
                status="passed",
                final_failure_reasons=[],
                noise_detected_before=False,
                noise_detected_after=False,
            )
        slides_payload = assess_slides_payload_for_video(
            slides_payload,
            input_video=input_video,
            explicit_path=bool(getattr(args, "slides", None)),
        )
        effective_visual_source_mode = determine_visual_source_mode(
            requested_mode=str(getattr(args, "visual_source_mode", "auto")),
            slides_payload=slides_payload,
        )
        slides_payload["requested_visual_source_mode"] = str(getattr(args, "visual_source_mode", "auto"))
        slides_payload["effective_visual_source_mode"] = effective_visual_source_mode
        slides_index_path.write_text(json.dumps(slides_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        effective_visual_source_mode = determine_visual_source_mode(
            requested_mode=str(getattr(args, "visual_source_mode", "auto")),
            slides_payload={},
        )

    visual_alignment_path = Path(output_paths["visual_alignment_json"])
    ppt_alignment_debug_path = Path(output_paths["ppt_alignment_debug_md"])
    visual_alignment_payload: dict[str, Any] = {}
    should_refresh_alignment = (
        stage_plan["frames"]["run"]
        or stage_plan["ocr"]["run"]
        or stage_plan["review_artifacts"]["run"]
        or stage_plan["codex_review"]["run"]
        or not visual_alignment_path.exists()
    )
    if should_refresh_alignment:
        visual_units_payload = _load_json_payload(Path(output_paths["visual_units_json"]))
        cleaned_segments = _load_cleaned_segments(Path(output_paths["transcript_cleaned_txt"]))
        visual_alignment_payload = build_visual_alignment(
            review_segments=review_segments_payload.get("segments", []),
            segment_reviews=corrections_payload.get("segment_reviews", []),
            cleaned_segments=cleaned_segments,
            visual_units=visual_units_payload.get("units", []),
            slides=(
                slides_payload.get("slides", [])
                if effective_visual_source_mode == "slides-first" and bool(slides_payload.get("usable", False))
                else []
            ),
            frames=frames,
            visual_source_mode=effective_visual_source_mode,
        )
        visual_alignment_path.write_text(
            json.dumps(visual_alignment_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        ppt_alignment_debug_path.write_text(
            render_ppt_alignment_debug_markdown(
                title=input_video.stem,
                visual_alignment=visual_alignment_payload,
            ),
            encoding="utf-8",
        )
    else:
        visual_alignment_payload = _load_json_payload(visual_alignment_path)
        if visual_alignment_payload and not ppt_alignment_debug_path.exists():
            ppt_alignment_debug_path.write_text(
                render_ppt_alignment_debug_markdown(
                    title=input_video.stem,
                    visual_alignment=visual_alignment_payload,
                ),
                encoding="utf-8",
            )

    result = {
        "visual_alignment_payload": visual_alignment_payload,
        "slides_payload": slides_payload,
        "effective_visual_source_mode": effective_visual_source_mode,
        "slides_path": slides_path,
    }
    sync_pipeline_artifact_view(output_paths)
    return result


def run_structure_pipeline(
    context: dict[str, Any],
    alignment_result: dict[str, Any],
) -> dict[str, Any]:
    input_video = context["input_video"]
    output_paths = context["output_paths"]
    stage_plan = context["stage_plan"]
    visual_alignment_payload = alignment_result["visual_alignment_payload"]

    note_outline_path = Path(output_paths["note_outline_json"])
    note_blocks_path = Path(output_paths["note_blocks_json"])
    note_outline_payload: dict[str, Any] = {}
    note_blocks_payload: dict[str, Any] = {}
    should_refresh_alignment = (
        stage_plan["frames"]["run"]
        or stage_plan["ocr"]["run"]
        or stage_plan["review_artifacts"]["run"]
        or stage_plan["codex_review"]["run"]
        or not Path(output_paths["visual_alignment_json"]).exists()
    )
    should_refresh_note_structures = should_refresh_alignment or not note_outline_path.exists() or not note_blocks_path.exists()
    if should_refresh_note_structures:
        note_outline_payload = build_note_outline(title=input_video.stem, visual_alignment=visual_alignment_payload)
        note_blocks_payload = build_note_blocks(note_outline=note_outline_payload, visual_alignment=visual_alignment_payload)
        note_outline_path.write_text(json.dumps(note_outline_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        note_blocks_path.write_text(json.dumps(note_blocks_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        note_outline_payload = _load_json_payload(note_outline_path)
        note_blocks_payload = _load_json_payload(note_blocks_path)

    result = {
        "note_outline_payload": note_outline_payload,
        "note_blocks_payload": note_blocks_payload,
    }
    sync_pipeline_artifact_view(output_paths)
    return result


def run_note_pipeline(
    context: dict[str, Any],
    media_result: dict[str, Any],
    review_result: dict[str, Any],
    structure_result: dict[str, Any],
    alignment_result: dict[str, Any],
) -> dict[str, Any]:
    args = context["args"]
    input_video = context["input_video"]
    output_paths = context["output_paths"]
    duration_seconds = context["duration_seconds"]
    stage_plan = context["stage_plan"]
    frames = media_result["frames"]
    review_segments_payload = review_result["review_segments_payload"]
    corrections_payload = review_result["corrections_payload"]
    visual_alignment_payload = alignment_result["visual_alignment_payload"]
    note_outline_payload = structure_result["note_outline_payload"]
    note_blocks_payload = structure_result["note_blocks_payload"]

    note_generation_prompt_path = Path(output_paths["note_generation_prompt_md"])

    if stage_plan["note"]["run"]:
        review_segments_payload = _load_json_payload(Path(output_paths["review_segments_json"]))
        corrections_payload = _load_json_payload(Path(output_paths["transcript_corrections_json"]))
        visual_units_payload = _load_json_payload(Path(output_paths["visual_units_json"]))
        if not visual_alignment_payload:
            visual_alignment_payload = _load_json_payload(Path(output_paths["visual_alignment_json"]))
        if not note_outline_payload:
            note_outline_payload = _load_json_payload(Path(output_paths["note_outline_json"]))
        if not note_blocks_payload:
            note_blocks_payload = _load_json_payload(Path(output_paths["note_blocks_json"]))
        note_generation_prompt_path.write_text(
            build_note_generation_prompt(
                title=input_video.stem,
                source_video=input_video.name,
                note_outline=note_outline_payload,
                note_blocks=note_blocks_payload,
                corrections=corrections_payload.get("corrections", []),
            ),
            encoding="utf-8",
        )
        if not bool(getattr(args, "skip_codex_note", True)):
            require_binary("codex")
            run_codex_note_generation(output_paths, timeout_seconds=args.codex_timeout_seconds)
        else:
            transcript_excerpt = read_excerpt(
                preferred_transcript_path(
                    Path(output_paths["transcript_txt"]),
                    Path(output_paths["transcript_cleaned_txt"]),
                )
            )
            markdown = build_note_markdown(
                title=input_video.stem,
                source_video=input_video.name,
                duration_seconds=duration_seconds,
                transcript_excerpt=transcript_excerpt,
                corrections=corrections_payload.get("corrections", []),
                frames=frames,
                visual_alignment=visual_alignment_payload,
                note_outline=note_outline_payload,
                note_blocks=note_blocks_payload,
            )
            note_path = Path(output_paths["note_path"])
            note_path.write_text(markdown, encoding="utf-8")
            failure_reasons = explain_note_generation_failure(note_path)
            write_note_generation_report(
                output_paths,
                generator="rule_based",
                attempts=[
                    {
                        "attempt": 1,
                        "prompt_type": "rule_based_render",
                        "command_status": "completed" if not failure_reasons else "quality_gate_failed",
                        "failure_reasons": failure_reasons,
                        "detail_excerpt": "",
                        "quality_gate_passed": not failure_reasons,
                    }
                ],
                status="passed" if not failure_reasons else "failed",
                final_failure_reasons=failure_reasons,
            )
            if failure_reasons:
                raise RuntimeError(f"rule-based note generation failed quality gate: {'; '.join(failure_reasons)}")

    sync_pipeline_artifact_view(output_paths)
    return {}


def run_pipeline(args: Any) -> dict[str, Any]:
    context = _prepare_pipeline_context(args)
    media_result = run_media_pipeline(context)
    review_result = run_review_pipeline(context, media_result)
    alignment_result = run_alignment_pipeline(context, media_result, review_result)
    structure_result = run_structure_pipeline(context, alignment_result)
    run_note_pipeline(context, media_result, review_result, structure_result, alignment_result)
    sync_pipeline_artifact_view(context["output_paths"])

    output_paths = context["output_paths"]
    input_video = context["input_video"]
    duration_seconds = context["duration_seconds"]
    stage_plan = context["stage_plan"]
    slides_path = alignment_result["slides_path"]
    slides_payload = alignment_result["slides_payload"]
    effective_visual_source_mode = alignment_result["effective_visual_source_mode"]
    args = context["args"]

    write_metadata(
        output_paths=output_paths,
        input_video=input_video,
        duration_seconds=duration_seconds,
        frame_interval=args.frame_interval,
        max_frames=args.max_frames,
        whisper_model=args.whisper_model,
        language=args.language,
        visual_source_mode=effective_visual_source_mode,
        slides_path=slides_path,
        slides_usable=bool(slides_payload.get("usable", False)),
    )
    sync_pipeline_artifact_view(output_paths)

    return {
        "input_video": input_video,
        "output_paths": output_paths,
        "duration_seconds": duration_seconds,
        "stage_plan": stage_plan,
    }


def main() -> None:
    args = parse_args()
    result = run_pipeline(args)
    print(f"Generated note at {result['output_paths']['note_path']}")

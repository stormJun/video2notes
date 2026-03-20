# Resumable Single Video Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add stage-level resume/skip behavior for the single-video pipeline, with explicit `--force-*` overrides, while splitting orchestration and artifact-planning logic out of `video_to_notes.py`.

**Architecture:** Keep the existing CLI contract and output structure, but move output-path planning, artifact readiness checks, and stage-plan computation into small helper modules. The entrypoint computes a stage plan, skips completed stages by default, and reruns downstream stages when an upstream force flag invalidates their outputs.

**Tech Stack:** Python 3.12, pytest, ffmpeg/ffprobe, openai-whisper, tesseract OCR, Codex CLI

---

### Task 1: Add failing tests for artifact readiness helpers

**Files:**
- Modify: `tests/test_video_to_notes.py`
- Create: `video_to_notes_artifacts.py`

**Step 1: Write the failing test**

Add tests for:
- `audio_artifact_ready(...)`
- `transcript_artifacts_ready(...)`
- `frames_artifacts_ready(...)`
- `ocr_artifacts_ready(...)`
- `review_artifacts_ready(...)`
- `codex_review_ready(...)`

Use temp directories to cover both complete and incomplete artifact sets.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_video_to_notes.py -q -k 'artifact_ready or review_ready'`
Expected: FAIL because the helper module and functions do not exist.

**Step 3: Write minimal implementation**

Implement readiness helpers in `video_to_notes_artifacts.py` with narrow, test-driven completeness checks.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_video_to_notes.py -q -k 'artifact_ready or review_ready'`
Expected: PASS

### Task 2: Add failing tests for stage plan resolution

**Files:**
- Modify: `tests/test_video_to_notes.py`
- Modify: `video_to_notes_artifacts.py`

**Step 1: Write the failing test**

Add tests for `resolve_stage_plan(...)` that verify:
- completed artifacts cause stages to skip by default
- `--force-transcribe` invalidates `review_artifacts`, `codex_review`, and `note`
- `--force-frames` invalidates `ocr`, `review_artifacts`, `codex_review`, and `note`
- `metadata` is always marked to run

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_video_to_notes.py -q -k stage_plan`
Expected: FAIL because no stage-plan resolver exists.

**Step 3: Write minimal implementation**

Add:
- `expected_frame_count(...)`
- `resolve_stage_plan(...)`
- small data structure for stage decisions and skip reasons

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_video_to_notes.py -q -k stage_plan`
Expected: PASS

### Task 3: Add failing tests for resumable orchestration

**Files:**
- Modify: `tests/test_video_to_notes.py`
- Create: `video_to_notes_pipeline.py`

**Step 1: Write the failing test**

Add orchestration tests that monkeypatch side-effect functions and verify:
- when artifacts are already complete, audio/transcribe/frames/ocr are not called
- when `force_transcribe=True`, transcription and downstream stages rerun
- when `force_codex_review=True`, only review/note/metadata rerun

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_video_to_notes.py -q -k 'resumable_pipeline or force_transcribe or force_codex_review'`
Expected: FAIL because orchestration is still hard-coded in `video_to_notes.py`.

**Step 3: Write minimal implementation**

Move orchestration into `video_to_notes_pipeline.py`:
- accept parsed args
- compute output paths
- compute stage plan
- execute only required stages
- return execution details useful for tests

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_video_to_notes.py -q -k 'resumable_pipeline or force_transcribe or force_codex_review'`
Expected: PASS

### Task 4: Add failing tests for CLI force flags and preserve imports

**Files:**
- Modify: `tests/test_video_to_notes.py`
- Modify: `video_to_notes.py`

**Step 1: Write the failing test**

Extend CLI parsing tests to verify:
- all new `--force-*` flags exist
- defaults are `False`
- explicit flags parse to `True`

Also preserve existing imports used by tests from `video_to_notes.py`.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_video_to_notes.py -q -k 'parse_args.*force or force_'`
Expected: FAIL because the flags do not exist yet.

**Step 3: Write minimal implementation**

Update `video_to_notes.py` to:
- keep CLI parsing
- expose current public helpers needed by tests
- delegate runtime work to `video_to_notes_pipeline.py`

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_video_to_notes.py -q -k 'parse_args.*force or force_'`
Expected: PASS

### Task 5: Run full test suite and verify real resume behavior

**Files:**
- Modify: `video_to_notes.py`
- Modify: `video_to_notes_pipeline.py`
- Modify: `video_to_notes_artifacts.py`
- Modify: `tests/test_video_to_notes.py`

**Step 1: Run the full tests**

Run: `pytest tests/test_video_to_notes.py -q`
Expected: PASS

**Step 2: Run the verified sample command on a fresh output root**

Run: `python3 video_to_notes.py --input 'materials/videos/4.1BERT-2训练.mp4' --output-root work/_resume_verify_fresh --frame-interval 60 --max-frames 4 --whisper-model small --language zh`
Expected: full pipeline outputs including `review_report.json` and `note.md`

**Step 3: Run the same command again without force flags**

Run: `python3 video_to_notes.py --input 'materials/videos/4.1BERT-2训练.mp4' --output-root work/_resume_verify_fresh --frame-interval 60 --max-frames 4 --whisper-model small --language zh`
Expected: heavy stages are skipped; `metadata.json` refreshes; final outputs still exist

**Step 4: Run a forced downstream re-execution check**

Run: `python3 video_to_notes.py --input 'materials/videos/4.1BERT-2训练.mp4' --output-root work/_resume_verify_fresh --frame-interval 60 --max-frames 4 --whisper-model small --language zh --force-codex-review`
Expected: review and note rerun without forcing Whisper/audio/frame extraction

**Step 5: Fix only the failures discovered, then rerun verification**

If any verification fails, patch the minimal code path and rerun the exact failing command until the evidence matches the expected behavior.

# Single Video Notes Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a first-pass single-video pipeline that turns one MP4 into transcript files, extracted frames, OCR output, metadata, and a Markdown note.

**Architecture:** Use a single Python CLI as the orchestration layer. Keep side-effecting media/OCR/transcription steps thin and move naming, timestamp, note assembly, and frame selection into pure helper functions so they can be tested directly.

**Tech Stack:** Python 3.12, pytest, ffmpeg/ffprobe, openai-whisper, tesseract OCR

---

### Task 1: Create tested helper functions

**Files:**
- Create: `tests/test_video_to_notes.py`
- Create: `video_to_notes.py`

**Step 1: Write the failing tests**

Add tests for:
- safe output directory naming from a video path
- seconds-to-timestamp formatting
- note assembly from metadata, transcript excerpts, and frame entries

**Step 2: Run tests to verify they fail**

Run:

```bash
cd /Users/songxijun/workspace/transformer_downloader
pytest tests/test_video_to_notes.py -q
```

Expected: fail because `video_to_notes.py` and required functions do not exist.

**Step 3: Write minimal implementation**

Implement pure helpers:
- `slugify_stem(path_str)`
- `format_seconds(seconds)`
- `build_note_markdown(...)`

**Step 4: Run tests to verify they pass**

Run:

```bash
cd /Users/songxijun/workspace/transformer_downloader
pytest tests/test_video_to_notes.py -q
```

Expected: all tests pass.

### Task 2: Add orchestration CLI

**Files:**
- Modify: `video_to_notes.py`

**Step 1: Add a failing CLI-oriented test**

Test a small function that resolves the output structure for a single input video and returns:
- output directory
- transcript file paths
- frame directory path
- note path

**Step 2: Run test to verify it fails**

Run:

```bash
cd /Users/songxijun/workspace/transformer_downloader
pytest tests/test_video_to_notes.py -q
```

Expected: fail because the output planning helper is missing.

**Step 3: Write minimal implementation**

Add:
- argparse CLI
- `plan_output_paths(...)`
- shell-out wrappers for ffmpeg, whisper, and tesseract
- basic frame extraction on a fixed interval
- JSON metadata and OCR file writing

**Step 4: Run tests to verify they pass**

Run:

```bash
cd /Users/songxijun/workspace/transformer_downloader
pytest tests/test_video_to_notes.py -q
```

Expected: all tests pass.

### Task 3: Verify with one real video

**Files:**
- Modify: `video_to_notes.py` if fixes are needed

**Step 1: Run the real pipeline**

Run:

```bash
cd /Users/songxijun/workspace/transformer_downloader
python3 video_to_notes.py \
  --input '4.1BERT-2训练.mp4' \
  --output-root notes \
  --frame-interval 60 \
  --max-frames 4 \
  --whisper-model tiny \
  --language zh
```

Expected outputs:
- `notes/4-1bert-2训练/metadata.json`
- `notes/4-1bert-2训练/transcript.txt`
- `notes/4-1bert-2训练/transcript.srt`
- `notes/4-1bert-2训练/ocr.json`
- `notes/4-1bert-2训练/assets/*.jpg`
- `notes/4-1bert-2训练/note.md`

**Step 2: Inspect outputs**

Confirm:
- transcript files are non-empty
- at least one frame exists
- note references local image files

**Step 3: Fix minimal issues**

Only fix real failures discovered in the run.

### Task 4: Document usage

**Files:**
- Modify: `PROJECT_SPEC.md`

**Step 1: Add a short execution note**

Document:
- required tools
- one sample command
- expected output structure

**Step 2: Verify command still works**

Re-run the sample command if the script changed.

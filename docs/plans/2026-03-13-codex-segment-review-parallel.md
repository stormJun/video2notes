# Codex Segment Review Parallelism Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Execute Codex transcript review per segment with bounded parallelism, then merge the segment results back into the existing review artifacts.

**Architecture:** Keep the current top-level CLI and quality gate, but replace the single review workspace with per-segment workspaces plus a bounded worker pool. Each segment keeps its own retry loop and reports, and the main thread merges successful segment results in the original segment order.

**Tech Stack:** Python 3, pytest, concurrent.futures, Codex CLI

---

### Task 1: Add failing tests for segment merge behavior

**Files:**
- Modify: `tests/test_video_to_notes.py`
- Test: `tests/test_video_to_notes.py`

**Step 1: Write the failing test**

Add a test that simulates two segment review jobs finishing out of order and verifies the merged `transcript.cleaned.txt` and merged `segment_reviews` still follow `review_segments.json` order.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_video_to_notes.py -q -k merge_segment_review_results`
Expected: FAIL because there is no segment merge implementation yet.

### Task 2: Add failing tests for bounded parallel retry behavior

**Files:**
- Modify: `tests/test_video_to_notes.py`
- Test: `tests/test_video_to_notes.py`

**Step 1: Write the failing test**

Add a test that simulates three segments where one passes immediately, one needs a JSON-only retry, and one fails after retry. Verify only failed segments retry and the final report records segment-level attempts.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_video_to_notes.py -q -k segment_retry`
Expected: FAIL because the current review flow only retries the whole transcript.

### Task 3: Implement segment review helpers

**Files:**
- Modify: `video_to_notes.py`
- Test: `tests/test_video_to_notes.py`

**Step 1: Write minimal implementation**

Add pure or mostly isolated helpers for:
- segment workspace path planning
- segment prompt construction
- segment result normalization
- final merge of cleaned text and corrections payloads

**Step 2: Run targeted tests**

Run: `pytest tests/test_video_to_notes.py -q -k 'merge_segment_review_results or segment_retry'`
Expected: PASS

### Task 4: Replace the whole-review runner with bounded parallel segment execution

**Files:**
- Modify: `video_to_notes.py`
- Test: `tests/test_video_to_notes.py`

**Step 1: Add a failing integration-style unit test**

Add a test that patches the Codex command runner, runs `run_codex_review(...)`, and verifies:
- one Codex job is launched per segment
- `--codex-review-parallelism` is honored
- the merged outputs are written to the top-level artifact paths

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_video_to_notes.py -q -k parallelism`
Expected: FAIL because `run_codex_review` still launches one whole-review job.

**Step 3: Write minimal implementation**

Update `run_codex_review(...)` to:
- read `review_segments.json`
- schedule per-segment review jobs with bounded parallelism
- retry only failed segments
- merge results
- write the final `review_report.json`

**Step 4: Run targeted tests**

Run: `pytest tests/test_video_to_notes.py -q -k 'parallelism or merge_segment_review_results or segment_retry'`
Expected: PASS

### Task 5: Add CLI plumbing and full verification

**Files:**
- Modify: `video_to_notes.py`
- Modify: `tests/test_video_to_notes.py`

**Step 1: Add a parsing test**

Extend CLI parsing coverage to verify `--codex-review-parallelism` defaults to `2` and accepts explicit overrides.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_video_to_notes.py -q -k codex_review_parallelism`
Expected: FAIL because the CLI flag is missing.

**Step 3: Write minimal implementation**

Add the CLI argument and thread it into `run_codex_review(...)`.

**Step 4: Run full tests**

Run: `pytest tests/test_video_to_notes.py -q`
Expected: PASS

**Step 5: Re-run the verified sample command**

Run: `python3 video_to_notes.py --input 'materials/videos/4.1BERT-2训练.mp4' --output-root notes --frame-interval 60 --max-frames 4 --whisper-model small --language zh`
Expected: the sample directory contains `review_segments.json`, `transcript.cleaned.txt`, `transcript.corrections.json`, `review_report.json`, and `note.md`.

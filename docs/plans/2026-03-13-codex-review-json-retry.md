# Codex Review JSON Retry Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** When the first review attempt already produced a usable `transcript.cleaned.txt`, make the retry focus on filling `transcript.corrections.json` instead of redoing the whole review.

**Architecture:** Keep the existing isolated review workspace and quality gate. Add a narrower retry prompt path that is selected only when the remaining failure reasons are JSON-only, so the second attempt has less work and a better chance of finishing before timeout.

**Tech Stack:** Python, pytest, Codex CLI prompt generation

---

### Task 1: Lock the JSON-only retry behavior with tests

**Files:**
- Modify: `tests/test_video_to_notes.py`
- Test: `tests/test_video_to_notes.py`

**Step 1: Write the failing test**

Add a test that simulates a first attempt which updates only `transcript.cleaned.txt`, leaves `transcript.corrections.json` pending, and verifies the retry prompt tells Codex to keep the cleaned transcript and only complete the JSON review fields.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_video_to_notes.py -q -k json`
Expected: FAIL because the current retry prompt still asks for the full review flow.

### Task 2: Implement the minimal JSON-only retry path

**Files:**
- Modify: `video_to_notes.py`
- Test: `tests/test_video_to_notes.py`

**Step 1: Write minimal implementation**

Add a helper that detects "cleaned transcript is already good, only JSON gates remain" and builds a narrower retry prompt for that case. Use it from `run_codex_review` without changing the existing pass/fail rules.

**Step 2: Run targeted tests**

Run: `pytest tests/test_video_to_notes.py -q -k 'json or retries_once_with_failure_feedback'`
Expected: PASS

**Step 3: Run full verification**

Run: `pytest tests/test_video_to_notes.py -q`
Expected: PASS

**Step 4: Re-run short smoke**

Run: `python3 video_to_notes.py --input 'materials/videos/1.1.4LSTM-1基本原理.mp4' --output-root 'work/short_smoke_20260313_v5' --frame-interval 60 --max-frames 4 --whisper-model small --language zh --codex-timeout-seconds 120`
Expected: `review_report.json` passes and `note.md` is generated.

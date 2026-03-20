# Codex Segment Review Design

**Date:** 2026-03-13

**Goal:** Replace the current whole-transcript Codex review flow with segment-scoped review jobs that can run with bounded parallelism, while keeping the existing output schema and quality gate intact.

## Context

The current implementation writes `review_segments.json`, then runs one or two whole-workspace `codex exec` calls against the entire transcript. This makes failures expensive because one missing JSON field or timeout can force a full retry. It also limits future concurrency because the unit of work is the entire transcript.

The approved direction is:

- each review segment gets its own isolated Codex workspace
- the main process can run a bounded number of those segment jobs in parallel
- each segment persists its own cleaned text and correction payload
- the main process merges segment outputs into the final `transcript.cleaned.txt`, `transcript.corrections.json`, and `review_report.json`

## Non-Goals

- Changing the top-level note generation format
- Changing the fixed schema of `transcript.corrections.json`
- Introducing unbounded parallelism
- Replacing Codex review with static term dictionaries

## Design

### 1. Segment workspaces

For each entry in `review_segments.json`, create a dedicated workspace under a review temp root. Each workspace contains only the files needed for that segment:

- `segment_input.json`
- `segment_transcript.txt`
- `segment.cleaned.txt`
- `segment.corrections.json`
- `codex_review_prompt.md`
- `AGENTS.md`
- optional `ocr.json`
- optional `assets/`

This keeps each `codex exec` prompt small and makes retries local to the failed segment.

### 2. Segment prompts and payloads

Each segment prompt instructs Codex to:

- read only the segment transcript and that segment's metadata
- write the corrected text for that segment to `segment.cleaned.txt`
- write segment-scoped review results to `segment.corrections.json`
- preserve the fixed correction entry schema

The segment correction payload will still use the global keys:

- `review_status`
- `last_updated`
- `segment_reviews`
- `corrections`

But `segment_reviews` will contain exactly one completed entry for the current segment.

### 3. Bounded parallel scheduler

The pipeline adds `--codex-review-parallelism` with default `2`.

- `1` means serial segment review
- `2` or `3` allows bounded parallel execution
- the scheduler submits one Codex job per segment and never exceeds the configured worker count

The implementation should avoid depending on output arrival order. Merge order always follows the original `review_segments.json` sequence.

### 4. Segment retries and failure handling

Each segment has its own retry loop:

- first attempt runs the full segment review prompt
- if the cleaned segment looks usable but JSON is incomplete, retry with a JSON-only prompt
- otherwise retry with a segment-scoped failure-feedback prompt
- retries never re-run successful segments

The whole review step passes only if every segment passes its quality gate.

### 5. Merge rules

After all segment jobs finish:

- `transcript.cleaned.txt` is built by concatenating `segment.cleaned.txt` in segment order
- `transcript.corrections.json.segment_reviews` is the ordered concatenation of each segment's single review entry
- `transcript.corrections.json.corrections` is the ordered concatenation of all segment corrections
- `review_status` becomes `done` only if every segment passed
- `last_updated` uses the merge timestamp when the merged payload is complete

### 6. Reporting

`review_report.json` should retain the current top-level summary and add useful segment-level visibility:

- per-segment attempt count
- per-segment pass/fail state
- per-segment failure reasons
- aggregate counts for planned, passed, and failed segments

## Testing Strategy

The change is test-first and should cover:

- segment success merging into a stable final transcript
- merge order staying aligned with `review_segments.json` even when jobs finish out of order
- retrying only failed segments
- switching a failed segment to a JSON-only retry when the cleaned text already passes
- generating a failed review report with segment-level details

## Risks

- Real `codex exec` concurrency may hit provider-side limits
- Segment prompts that are too small may lose useful global context
- Merge bugs could silently corrupt final transcript ordering

These are mitigated by:

- keeping concurrency bounded and configurable
- including OCR and nearby frame hints per segment
- enforcing merge-order tests and output quality gates

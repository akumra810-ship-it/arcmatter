# arcmatter
ARCMATTER
# Video Labeling Platform — Design Document

## 1. Product Summary

A Labelbox-style platform for **video** labeling, purpose-built for robotics training data. Clients upload videos of any length. An AI model (client's choice: open-source SLM/LLM like DeepSeek, Qwen, Kimi, or closed-source like Anthropic's Claude) runs object detection + activity description on the video while it plays. Human labellers can be assigned to label from scratch or verify/correct AI output, frame-by-frame, with free-text descriptions of what's happening at precise timestamps. The system tracks labeller agreement/disagreement with the AI over time, and produces a final structured JSON export for robotics training pipelines.

---

## 2. Existing Code Assessment (carried over)

The provided C++/pybind11 module is a single-model (YOLO ONNX via OpenCV DNN) async detector with basic IOU tracking. It is **not reusable as the core inference engine** for this product because:

- It assumes a fixed local ONNX model with 80 hardcoded COCO classes — incompatible with open-vocabulary, prompt-driven VLMs (Claude, Qwen-VL, DeepSeek-VL, Kimi).
- It has no concept of pluggable model providers, API calls, or prompts.
- It has no persistence layer, auth, human workflow, or export format.
- It has threading/lifecycle bugs (detached threads with no cleanup, non-unique job IDs, no cancellation).

**What's worth keeping conceptually:**
- The async job pattern (submit → poll status → fetch results) is the right shape for long-running video processing and should be preserved in the new architecture (as a job queue, not a raw detached thread).
- The idea of a tracker running as a *second pass* over raw per-frame detections (to assign consistent track IDs) is reusable — but it should run after any model's output, not just YOLO's.

Everything else should be rebuilt. Full design below.

---

## 3. High-Level Architecture

```
┌─────────────┐      ┌──────────────────┐      ┌────────────────────┐
│   Web App    │◄────►│    API Gateway    │◄────►│   Auth / Org / RBAC │
│ (React/Next) │      │   (REST/GraphQL)  │      └────────────────────┘
└─────┬───────┘      └────────┬─────────┘
      │  video playback+labels │
      │                        ▼
      │              ┌──────────────────┐
      │              │  Task Orchestrator │  (job queue: SQS/Redis/Temporal)
      │              └───────┬──────────┘
      │                      │
      │        ┌─────────────┼─────────────────┐
      │        ▼             ▼                 ▼
      │  ┌───────────┐ ┌────────────┐   ┌──────────────┐
      │  │  Frame     │ │  Model      │   │  Human Task   │
      │  │  Extractor │ │  Inference   │   │  Queue        │
      │  │ (ffmpeg)   │ │  Workers     │   │  (assignment) │
      │  └─────┬─────┘ └─────┬──────┘   └───────┬──────┘
      │        │             │                  │
      │        ▼             ▼                  ▼
      │  ┌─────────────────────────────────────────┐
      │  │        Object Storage (S3/GCS)            │
      │  │   raw video, frames, thumbnails            │
      │  └─────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────┐
│   Primary DB (Postgres) — jobs, labels, tracks,   │
│   corrections, users, model runs, QA history       │
└─────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Export Service       │──► Final JSON (per video) → Robotics training pipeline
└─────────────────────┘
```

---

## 4. Functional Requirements by Module

### 4.1 Video Ingestion

- Accept upload of videos of **arbitrary length** via chunked/resumable upload (e.g., tus protocol or S3 multipart) — do not rely on a single HTTP request body for large files.
- On upload completion, trigger a transcode/normalize step (ffmpeg) to:
  - Normalize to constant frame rate (CFR) — critical, since the existing code's frame-index assumption breaks on variable frame rate (VFR) source video, which is extremely common (phone footage, screen recordings).
  - Generate a low-res proxy for smooth scrubbing/playback in the browser, while keeping the original for final inference/export.
  - Extract metadata (duration, fps, resolution, codec) into the DB.
- Store raw + proxy + frame extracts in object storage, never in the app DB.
- Support long videos by **chunking into segments** (e.g., 60–120s segments) for both inference and UI virtualization — never load a whole 2-hour video's frames into memory/DOM at once.

### 4.2 Model Selection & Inference Orchestration

- Client selects a model per job from a registry of providers:
  - **Closed-source API**: Anthropic (Claude), etc. — called via HTTP with API key, images sent as base64 frames + a structured prompt requesting JSON-formatted detections/descriptions.
  - **Open-source SLM/LLM**: DeepSeek-VL, Qwen-VL, Kimi-VL — either called via a hosted API or self-hosted inference server (vLLM / TGI / Ollama) behind an internal endpoint with the same interface contract as the closed-source path.
- Build a **Model Provider abstraction** so every backend implements the same interface regardless of vendor:
  ```
  interface ModelProvider {
    name: string
    supportsBatching: boolean
    async detectFrame(frame: Image, prompt: PromptConfig): DetectionResult[]
  }
  ```
  This lets you add/remove models without touching the orchestrator, and lets clients A/B two models on the same video later if desired.
- Prompting strategy for VLMs: since these are not fixed-class detectors like YOLO, define a **structured output contract** (ask the model to return strict JSON: bounding boxes normalized 0–1, label, short description, confidence) and validate/reject malformed responses with retry.
- **Parallelism while video plays**: this is the key UX requirement, and needs distinguishing between two things clients might mean:
  1. *Frames are inferred out-of-order, in parallel, ahead of playback* — a pool of workers pulls frames from a queue and runs inference concurrently, writing results back keyed by frame index/timestamp; the player polls/subscribes for results as they land, so labels "catch up" to and eventually outpace playback. This is realistic and is what should be built.
  2. *Inference happens with zero latency exactly in sync with playback position* — not realistic for VLM API calls (100ms–2s+ per frame depending on model), so don't promise this; instead show a "processing" state per segment and progressively reveal labels, similar to YouTube's caption-generation UX.
- Use a real task queue (Redis+BullMQ, SQS, or Temporal — Temporal is worth it here for the retry/human-approval workflow later) rather than a detached thread and in-memory map. Jobs must survive process restarts.
- Sample rate is configurable per job (e.g., every Nth frame or every X ms) to control cost — VLM API calls are expensive per-frame; don't run every single frame at high fps by default.
- Post-inference tracking pass: reuse the IOU/track-ID association concept, but move it to a service that runs against **any** provider's normalized output, so track continuity works the same regardless of which model produced the boxes. Consider upgrading beyond IOU-only tracking (e.g., adding motion prediction) if objects move fast or occlude often — flag as a v2 improvement, not blocking for v1.

### 4.3 Frontend: Video Player & Labeling UI

- Smooth scrub/playback: use proxy video + a virtualized timeline component; don't re-render the whole DOM per frame.
- Timeline UI needs:
  - Bounding box overlay synced to current playback time, sourced from stored detections (interpolated between sampled frames if sampling rate < video fps — this is exactly where the existing `interpolate_keyframes` logic is legitimately reusable, generalized to work off any provider's boxes, not just YOLO's).
  - A frame-precise (not just second-precise) scrubber — internally key all annotations by **timestamp in milliseconds**, not frame index, since frame index is meaningless once you mix original video, proxy video, and variable sampling rates.
  - Click-to-add free-text annotation pinned to an exact timestamp (e.g., the "arc welding of 2 joints..." example) — stored as a separate annotation type from bounding boxes, but both anchored to the same timeline.
  - Support overlapping/duration-based annotations too (e.g., "job starts" at 1:31 through "job ends" at 4:02), not just single-point-in-time notes, since real work descriptions span a range.
- Labeling modes:
  - **Fresh labeling**: human draws boxes + writes descriptions from scratch (no AI involved).
  - **Verification**: human reviews AI-generated boxes/labels overlaid on the video, and for each detection (or each frame's detection set) clicks **Correct** or **Incorrect**. Incorrect triggers an inline correction UI (adjust box, relabel class, edit description).
- All UI interactions should autosave continuously (don't rely on a manual "save" button for long labeling sessions).

### 4.4 Human-in-the-Loop Workflow

- **Task types**: `label_from_scratch`, `verify_ai_output`, `qa_review` (review another human's work).
- **Task assignment**: a queue/marketplace model — tasks can be assigned to a specific labeller or pulled from a pool; support priority and deadline fields.
- **Correct/Incorrect tracking with history**:
  - Every AI-generated detection gets a `verification_status` (`unverified`, `correct`, `incorrect`) plus a `verified_by` and `verified_at`.
  - When marked incorrect, do not overwrite — create a **correction record** linked to the original detection, preserving the original AI output immutably. This gives you a full audit trail: what the model said, what the human changed it to, who changed it, when.
  - Aggregate this over time per model/version to compute **accuracy metrics** (e.g., % correct per class, per video type, drift over time as model versions change) — this is valuable both for picking which model to default clients to, and as a QA dashboard for you internally.
- **Reviewer/QA layer**: optionally require a second human to review a first labeller's corrections before finalizing (standard double-review pattern from data-labeling pipelines) — configurable per project, not mandatory for all.
- **Inter-annotator agreement**: if you ever have two humans label the same clip, compute IOU-based agreement between them as an additional quality signal — worth designing the data model to support even if not built in v1.

### 4.5 Export / Final Report

- Output is a **per-video JSON** (schema below) combining:
  - Final (human-verified, or AI-only if no human involved) bounding boxes + track IDs over time.
  - Frame/time-range text annotations describing actions.
  - Provenance: which model produced each detection, whether/how a human corrected it.
- Export should be versioned and re-generatable at any time (don't treat it as a one-time destructive action) — a job can be "exported as of state X" while labeling continues.
- Provide both a full-fidelity JSON and (optionally) a more robotics-pipeline-friendly derivative — e.g., COCO-video format or a custom action-segmented format — since "AI robotics labs" consuming this may expect specific conventions (check with them on ingestion format early, this is a real integration risk if assumed rather than confirmed).

---

## 5. Example Final JSON Schema

```json
{
  "video_id": "vid_8f2a1c",
  "source_filename": "welding_station_03.mp4",
  "duration_ms": 305000,
  "fps": 30,
  "resolution": { "width": 1920, "height": 1080 },
  "model_runs": [
    {
      "model_run_id": "run_001",
      "provider": "anthropic",
      "model": "claude-sonnet-5",
      "sample_interval_ms": 200,
      "started_at": "2026-09-10T14:02:00Z",
      "completed_at": "2026-09-10T14:09:12Z"
    }
  ],
  "tracks": [
    {
      "track_id": "track_1",
      "class_label": "welding_torch",
      "detections": [
        {
          "timestamp_ms": 91000,
          "bbox": { "x": 0.42, "y": 0.31, "w": 0.08, "h": 0.15 },
          "confidence": 0.94,
          "model_run_id": "run_001",
          "verification_status": "correct",
          "verified_by": "user_552",
          "verified_at": "2026-09-11T09:15:00Z",
          "correction_history": []
        },
        {
          "timestamp_ms": 91200,
          "bbox": { "x": 0.43, "y": 0.31, "w": 0.08, "h": 0.15 },
          "confidence": 0.61,
          "model_run_id": "run_001",
          "verification_status": "incorrect",
          "verified_by": "user_552",
          "verified_at": "2026-09-11T09:15:20Z",
          "correction_history": [
            {
              "corrected_bbox": { "x": 0.44, "y": 0.33, "w": 0.07, "h": 0.14 },
              "corrected_label": "welding_torch",
              "corrected_by": "user_552",
              "corrected_at": "2026-09-11T09:15:20Z",
              "note": "box drifted onto glove, tightened to torch tip"
            }
          ]
        }
      ]
    }
  ],
  "text_annotations": [
    {
      "annotation_id": "ann_01",
      "start_ms": 91000,
      "end_ms": 130000,
      "text": "Arc welding of 2 joints with left hand as support and right hand using MIG welding machine - job starts",
      "author": "user_552",
      "author_role": "human_labeller",
      "created_at": "2026-09-11T09:16:00Z"
    }
  ],
  "qa_summary": {
    "total_ai_detections": 4820,
    "verified_correct": 4410,
    "verified_incorrect": 298,
    "unverified": 112,
    "accuracy_estimate": 0.937
  }
}
```

---

## 6. Data Model (Core Tables)

- `organizations`, `users`, `roles` — auth/RBAC (client admin, human labeller, QA reviewer, robotics-lab viewer).
- `videos` — storage paths (raw, proxy), duration, fps, resolution, status.
- `jobs` — one per (video, model or task type) processing run; status, progress, queue metadata.
- `model_runs` — provider, model name/version, prompt config, timing — this is what lets you track accuracy drift by model version over time.
- `detections` — timestamp_ms, bbox, class_label, confidence, model_run_id, track_id, verification_status.
- `corrections` — FK to detection, corrected fields, corrected_by, corrected_at, note. Append-only.
- `tracks` — track_id, video_id, class_label, first/last seen timestamp.
- `text_annotations` — video_id, start_ms, end_ms (nullable for point-in-time), text, author, role.
- `tasks` — assignment records: task_type, assignee, status, deadline, video_id.
- `exports` — export_id, video_id, generated_at, generated_by, storage path to JSON.

---

## 7. Non-Functional Requirements

- **Scalability**: inference workers must scale horizontally and independently from the API/web tier; use a real queue, not in-process threads.
- **Cost control**: VLM API calls are the dominant cost — build in per-job budget caps, configurable sampling rates, and caching (don't re-run inference if a job is re-opened without changes).
- **Latency**: target first-detections-visible within a few seconds of upload+job start; full-video processing time scales with length and sampling rate — surface progress (`processed/total`), not just a spinner.
- **Reliability**: all long-running work must be resumable after a worker crash (idempotent frame processing, checkpointed progress) — this directly fixes the original code's "detached thread, no recovery" flaw.
- **Data integrity**: corrections are append-only/immutable for audit purposes; never overwrite raw AI output in place.
- **Security**: signed URLs for video/frame storage access, per-org data isolation, API keys for model providers stored server-side only (never exposed to the browser).
- **Variable frame rate handling**: normalize all uploaded video to CFR at ingestion time, and key all annotations by millisecond timestamp rather than frame index throughout the system.

---

## 8. Suggested Tech Stack

- **Frontend**: React/Next.js, a canvas or WebGL overlay library for bounding boxes (e.g., Konva or a custom canvas layer) on top of `<video>`, virtualized timeline component.
- **Backend API**: Node/TypeScript or Python (FastAPI) — either fine; FastAPI is a natural fit if your inference workers are also Python.
- **Task orchestration**: Temporal (recommended for the multi-step, retry-heavy, human-in-the-loop workflow) or Redis+BullMQ for a lighter-weight start.
- **Inference workers**: Python, calling model provider APIs; self-hosted open-source models served via vLLM behind an internal API matching the same provider interface.
- **DB**: Postgres (relational integrity matters here — corrections, tasks, QA history are all relational).
- **Object storage**: S3 or GCS for video/frames/exports.
- **Video processing**: ffmpeg for transcode/CFR-normalization/frame extraction.

---

## 9. Phased Build Plan

1. **Phase 1 — Core pipeline**: upload → CFR normalize → single model provider (start with Anthropic, simplest to integrate) → async inference queue → store detections → basic playback with box overlay. No human workflow yet.
2. **Phase 2 — Human verification**: correct/incorrect UI, correction history, task assignment, text annotations at timestamps.
3. **Phase 3 — Multi-model support**: add DeepSeek/Qwen/Kimi providers behind the abstraction, client-facing model picker, per-model QA accuracy dashboard.
4. **Phase 4 — Export & scale**: final JSON export service, budget/cost controls, horizontal scaling of workers, inter-annotator agreement if double-review is needed.

---

## 10. Open Questions to Resolve Before Building

- What exact JSON/format convention do the downstream robotics labs expect (custom vs. an existing standard like COCO-video or a robot-learning-specific format)? This should be confirmed with them, not assumed.
- What's the expected video length distribution (minutes vs. hours)? This changes chunking/storage cost assumptions significantly.
- Do clients need real-time (near-live) labeling of a video as it's being recorded/streamed, or is "upload then process" always acceptable? The requirements as written describe the latter, but worth confirming explicitly.
- What's the tolerance for AI labeling cost per video (this determines default sampling rate and which models are offered by default)?

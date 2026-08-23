# NextHire — Voice-Based AI Interview Screening Round (Prototype)

A LangGraph-based AI interviewer that reads a candidate's resume and a
company's hiring policy, conducts a live **voice** screening interview
(speech-to-text in, text-to-speech out), evaluates the candidate, and
generates a PDF report — with every step checkpointed to Postgres so a
crash mid-interview doesn't lose progress.

This is the screening-round slice of NextHire, a larger planned platform
(HR screening → technical rounds → behavioral → hiring manager). Built
to be extended one node at a time.

---

## How it works

```
START
  │
  ▼
load_resume_and_policy   reads resume + hiring-policy PDFs from Folder/Input/,
  │                      summarizes the resume, and extracts the policy into
  │                      a structured plan (round name, guide, questions,
  │                      question count — counted via a dedicated LLM call)
  ▼
screening                asks each required screening question out loud
  │                      (Kokoro TTS), listens for and transcribes the
  │                      candidate's spoken answer (Faster-Whisper STT),
  │                      and stores every Q&A pair in history — later
  │                      questions are grounded in the resume, the policy's
  │                      guide, and everything already asked (no repeats)
  ▼
final                    LLM evaluates the candidate against 10 criteria
  │                      (communication, technical understanding, resume
  │                      credibility, role fit, etc.) using the resume,
  │                      policy, and full interview transcript
  ▼
generate_pdf             builds a PDF report (resume summary + policy +
  │                      full transcript + evaluation) and saves it to
  │                      Folder/Output/
  ▼
END
```

Every node's output is checkpointed to Postgres as it completes, keyed
by a `thread_id` (currently the candidate's name) — if the process
crashes or is stopped, re-running with the same `thread_id` resumes
from the last completed node instead of starting over.

---

## Project structure

```
.
├── graph.py               # LangGraph nodes + state schema
├── main.py                 # entry point: preloads voice models, opens the
│                            # Postgres connection, runs the graph
├── voice_utils.py           # STT (Faster-Whisper) + TTS (Kokoro) helpers
├── requirements.txt
├── .env                     # your API keys / DB URL (not committed)
└── Folder/
    ├── Input/                 # <- put the resume + policy PDFs here
    │     ├── <candidate>_resume.pdf
    │     └── xyz hiring policy.pdf
    └── Output/                 # <- generated evaluation report PDFs land here
```

---

## Setup

### 1. System dependency

Kokoro (TTS) needs `espeak-ng` installed at the OS level (not via pip):

```bash
# Ubuntu / Debian / WSL
sudo apt-get update && sudo apt-get install -y espeak-ng

# macOS
brew install espeak-ng
```
Windows: install via the [espeak-ng releases page](https://github.com/espeak-ng/espeak-ng/releases), or run everything inside WSL instead — smoother for audio ML libraries on Windows generally.

### 2. Python environment

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
Use Python 3.9–3.12 — some ML packages here lag behind on brand-new Python versions.

### 3. Environment variables

Create a `.env` file in the project root:

```bash
# LLM
COHERE_API_KEY=your-cohere-key

# Postgres checkpointing (Render Postgres — use the "External Database URL")
DATABASE_URL=postgres://user:password@host:5432/dbname

# Optional: LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langsmith-key
LANGCHAIN_PROJECT=nexthire
```

| Variable | Required? | Notes |
|---|---|---|
| `COHERE_API_KEY` | Yes (recommended) | Without it, some prompts fall back to templated text instead of LLM-generated |
| `DATABASE_URL` | Yes | From Render Postgres dashboard. `main.py` auto-appends `sslmode=require` if missing — Render requires SSL |
| `LANGCHAIN_TRACING_V2` / `LANGCHAIN_API_KEY` / `LANGCHAIN_PROJECT` | No | Only needed if you want traces on the LangSmith dashboard |

### 4. Add your input files

Drop these two PDFs into `Folder/Input/`:
- The candidate's resume
- The hiring policy PDF (must include a numbered list of required screening questions and a guide for the round — that's what gets extracted and asked)

Update the filenames at the top of `graph.py` if they don't match:
```python
RESUME_PDF_PATH = os.path.join(INPUT_DIR, "NILARDRI_PRAMANICK_RESUME_v4.pdf")
POLICY_PDF_PATH = os.path.join(INPUT_DIR, "xyz hiring policy_3.pdf")
```

### 5. Run it

```bash
python main.py
```

You'll be asked for a candidate name, then the interview runs live:
each question is spoken out loud, you answer by voice (push-to-talk —
press Enter to start recording, Enter again to stop), and at the end
you'll see the full transcript, the evaluation, and the path to the
generated PDF report.

---

## Known limitations / what's next

- **Push-to-talk, not always-listening.** Simpler to get right first; a
  VAD-based (voice-activity-detection) auto-listen mode is a natural
  upgrade once this loop is solid.
- **CPU-only latency.** Expect ~2–6 seconds of dead air between the
  candidate finishing and the interviewer responding, on CPU. A GPU
  (even free-tier cloud) meaningfully cuts this down — see
  `voice_utils.py` for the `device="cuda"` swap.
- **One round only.** The graph currently ends after screening +
  evaluation. Adding technical/behavioral rounds is the same pattern:
  new node function, `builder.add_node(...)`, rewire the edge that
  currently points from `screening` to `final`.
- **Single-candidate CLI flow**, not yet a hosted multi-user platform —
  `thread_id` is currently just the candidate's name; a real deployment
  would need unique session IDs and a proper frontend instead of
  terminal input/output.

---

## Troubleshooting

- **Only 1 question gets asked instead of the number in the policy** —
  check the `[debug]` lines `load_resume_and_policy` prints; if the
  dedicated question-counting LLM call is under-counting, check that
  your policy PDF's questions are in a clear numbered list.
- **Transcriptions come out wrong / unrelated to what you said** —
  check the `peak amplitude` debug line printed after each recording;
  if it's below ~0.02, you're likely recording from the wrong
  microphone (`sd.query_devices()` shows all available devices).
- **Postgres connection fails** — make sure `sslmode=require` is on
  the connection string (Render requires SSL) and that you used the
  **External**, not Internal, database URL if running locally.
- **First run feels very slow** — that's mostly one-time model
  downloads/loading (Whisper + Kokoro weights). `main.py` preloads
  both before asking for a candidate name specifically so this cost is
  paid once, up front, not per question.

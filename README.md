# NextHire — Practice Interview with an AI Interviewer

NextHire is a voice-based AI interview practice tool. Upload your resume
and a question guide, and an AI interviewer asks you questions out loud,
listens to your spoken answers, evaluates you, and generates a PDF
report — all running locally on your own machine.

---
![Preview](./Preview.png)
---

## Features

- **Voice-based interview, not a chat box.** Questions are spoken aloud
  (Kokoro TTS) and your answers are recorded and transcribed (Faster-Whisper
  STT) — no typing required during the interview itself.
- **Upload your own resume and question guide.** No fixed company or role
  baked in — bring any resume and any PDF listing the questions you want
  to be asked, and NextHire builds the interview around them.
- **Grounded, non-repeating questions.** Each question is generated from
  your resume, the question guide, and everything already asked — the
  interviewer won't repeat itself.
- **Automatic evaluation.** After the round, an LLM scores you across 10
  criteria (communication, technical understanding, resume credibility,
  role fit, and more) with a brief justification for each.
- **PDF report, generated automatically.** A full report — resume summary,
  interview policy, transcript, and evaluation — is saved as a PDF after
  every session.
- **Two ways to run it:**
  - **Web app** (`backend.py` + `frontend.py`) — a browser-based UI where
    you upload files and talk to the interviewer through your mic.
  - **CLI** (`main.py`) — a terminal-based version using push-to-talk,
    reading fixed resume/policy PDFs from `Folder/Input/`.
- **Optional Postgres checkpointing.** The CLI flow can persist interview
  state to a database, so a crash mid-interview doesn't lose progress.

---

## Project structure

```
NextHire/
├── graph.py             LangGraph nodes: load_resume_and_policy, screening,
│                         final (evaluation), generate_pdf
├── api_helpers.py         Question-generation prompts, reused by the web backend
├── backend.py              FastAPI app (web mode): /start and /answer endpoints
├── frontend.py              Streamlit app (web mode): avatar + speaking/listening UI
├── main.py                  CLI entry point (terminal mode, with Postgres checkpointing)
├── voice_utils.py             STT (Faster-Whisper) + TTS (Kokoro) helpers
├── interviewer.png              Avatar shown in the web UI
├── requirements.txt
├── .env                          Your API keys / DB URL (not committed)
└── Folder/
    ├── Input/                      CLI mode reads its resume + policy PDFs here
    └── Output/                       Generated interview report PDFs land here
```

---

## Local setup

### 1. System dependency

Kokoro (TTS) needs `espeak-ng` installed at the OS level:

```bash
# Ubuntu / Debian / WSL
sudo apt-get update && sudo apt-get install -y espeak-ng

# macOS
brew install espeak-ng
```
Windows: install via the [espeak-ng releases page](https://github.com/espeak-ng/espeak-ng/releases), or run everything inside WSL.

### 2. Python environment

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
Use Python 3.9–3.12.

### 3. Environment variables

Create a `.env` file in the project root:

```bash
COHERE_API_KEY=your-cohere-key

# Only needed for the CLI's Postgres checkpointing (main.py)
DATABASE_URL=postgres://user:password@host:5432/dbname

# Optional: LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langsmith-key
LANGCHAIN_PROJECT=nexthire
```

---

## Running it

### Option A — Web app (recommended)

```bash
uvicorn backend:app --reload --port 8000        # terminal 1
streamlit run frontend.py                        # terminal 2
```

Open the Streamlit URL, enter your name, upload your resume and question
guide PDFs, and start — the avatar speaks each question, then switches to
listening for your recorded answer via your browser's microphone.

### Option B — CLI (terminal, push-to-talk)

Drop a resume and a policy PDF into `Folder/Input/`, matching the
filenames set in `graph.py`:
```python
RESUME_PDF_PATH = os.path.join(INPUT_DIR, "NILARDRI_PRAMANICK_RESUME_v4.pdf")
POLICY_PDF_PATH = os.path.join(INPUT_DIR, "xyz hiring policy_3.pdf")
```

Then run:
```bash
python main.py
```
You'll be asked for a candidate name, then the interview runs in the
terminal: each question is spoken, you answer by pressing Enter to
start/stop recording, and at the end you'll see the full transcript, the
evaluation, and the path to the generated PDF report.

---

## Known limitations

- Push-to-talk / manual recording, not always-listening (voice-activity
  detection).
- CPU-only inference means a few seconds of delay between your answer
  and the next question — a GPU noticeably speeds this up.
- Single screening round only — technical, behavioral, and hiring-manager
  rounds are a natural next step, following the same node pattern in
  `graph.py`.
- Runs locally; browser-based multi-user hosting requires a real deployment
  (see the web app's architecture notes if you want to take this further).

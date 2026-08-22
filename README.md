# Voice Interview Demo — start → intro_agent → end

A minimal, runnable example of one voice exchange in an AI interview:
the LangGraph agent generates an opening question, Kokoro speaks it,
you answer out loud, Faster-Whisper transcribes it.

## Files

- `graph.py` — the LangGraph graph (`START -> intro_agent -> END`)
- `voice_utils.py` — STT (Faster-Whisper) and TTS (Kokoro) helpers
- `main.py` — runs one full exchange end to end
- `requirements.txt` — Python dependencies

## 1. System dependency (required for Kokoro)

Kokoro needs `espeak-ng` installed on your system (not via pip).

**Ubuntu/Debian/WSL:**
```bash
sudo apt-get update && sudo apt-get install -y espeak-ng
```

**macOS:**
```bash
brew install espeak-ng
```

**Windows:** install via the [espeak-ng releases page](https://github.com/espeak-ng/espeak-ng/releases) (download the `.msi`), or run this inside WSL instead — WSL is the smoother path for audio ML libraries on Windows generally.

## 2. Python environment

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

If `pip install kokoro` or `faster-whisper` fails, make sure you're on Python 3.9–3.12 (not 3.13, some ML packages lag behind on new Python versions).

## 3. (Optional) Set your Cohere API key

Without this, `intro_agent` still works — it uses a templated fallback
question so you can test the voice pipeline immediately.

```bash
export COHERE_API_KEY="your-key-here"     # Windows: set COHERE_API_KEY=your-key-here
```

Get a free Cohere trial key at https://dashboard.cohere.com/api-keys if you don't have one.

## 4. Run it

```bash
python main.py
```

What happens:
1. It asks for a candidate name (just typed, for this demo).
2. `graph.py` runs `START -> intro_agent -> END` and generates the opening question as text.
3. `voice_utils.speak()` converts that text to audio with Kokoro and plays it through your speakers.
4. It waits for you to press Enter, records your spoken answer through your mic, and you press Enter again to stop.
5. `voice_utils.transcribe()` runs Faster-Whisper on the recording and prints the transcript.

## How this maps onto your real platform

Right now the graph has exactly one node (`intro_agent`) because that's
what you asked for. To extend it into your real HR screening / technical
round 1 / technical round 2 / behavioral / hiring manager flow:

- Add more nodes to `graph.py` (e.g. `hr_screening_agent`, `technical_round_1_agent`), each reading/writing to a shared `InterviewState`.
- Wire them with `builder.add_edge(...)` in the order rounds must run.
- After each `speak()` + `record_and_transcribe()` cycle in `main.py`, feed the transcribed answer back into `interview_graph.invoke(...)` with the updated state, so the next node picks up where the last one left off.
- Swap the push-to-talk `record_audio()` for a WebSocket + VAD-based version when you're ready to wire this into your FastAPI backend instead of running it as a local script — the STT/TTS functions themselves (`speak`, `transcribe`) don't need to change, only how audio gets in and out.

## Notes on performance

- First run downloads model weights (Whisper `base.en` is ~150MB, Kokoro is ~327MB) — this only happens once, they're cached afterward.
- CPU-only works fine for a demo; if you have a CUDA GPU, change `device="cpu"` to `device="cuda"` and `compute_type="int8"` to `compute_type="float16"` in `voice_utils.py` for faster transcription.
- Kokoro's `voice="af_heart"` is one of 54 built-in voices — browse the full list in the [Kokoro voices file](https://github.com/hexgrad/kokoro) if you want a different interviewer voice.

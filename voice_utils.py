"""
voice_utils.py
--------------
Everything voice-related, kept separate from the LangGraph logic so you
can swap STT/TTS backends later without touching graph.py.

Two directions of audio:
  - speak(text)          -> Kokoro generates audio, we play it out loud
  - record_and_transcribe() -> mic records candidate's answer (push-to-talk),
                                Faster-Whisper turns it into text

This uses PUSH-TO-TALK (press Enter to start, Enter again to stop) rather
than always-on VAD-based listening. It's simpler to get right first —
once this loop works end to end, you can swap record_and_transcribe()
for a VAD-driven streaming version.
"""

import threading
import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000  # Whisper expects 16kHz mono audio


# ---------------------------------------------------------------------
# STT: Faster-Whisper
# ---------------------------------------------------------------------

_whisper_model = None


def _get_whisper_model():
    """Lazy-load the model once, reuse it for every call."""
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        # "small.en" trades a bit of speed for meaningfully better accuracy
        # than "base.en" -- worth it for interview answers with varied
        # technical vocabulary. Use device="cuda", compute_type="float16"
        # if you have a GPU.
        _whisper_model = WhisperModel(
            "small.en", device="cpu", compute_type="int8"
        )
    return _whisper_model


def record_audio() -> np.ndarray:
    """
    Push-to-talk recording: press Enter to start recording, then press
    Enter again to stop. Returns a 1D float32 numpy array of audio.
    """
    print("\n>> Press Enter to start recording your answer...")
    input()
    print(">> Recording... press Enter again to stop.")

    frames = []
    stop_event = threading.Event()

    def callback(indata, frame_count, time_info, status):
        if not stop_event.is_set():
            frames.append(indata.copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=callback
    )
    with stream:
        input()  # blocks until Enter is pressed again
        stop_event.set()

    print(">> Recording stopped. Transcribing...")

    if not frames:
        return np.array([], dtype=np.float32)

    audio = np.concatenate(frames, axis=0).flatten()

    # Debug info: if peak is very low, you're likely recording from the
    # wrong device or the mic gain is too low -- Whisper will guess/
    # hallucinate on near-silent audio. If this prints something like
    # 0.001-0.01, that's your bug, not Whisper's accuracy.
    duration = len(audio) / SAMPLE_RATE
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    print(f">> Captured {duration:.1f}s of audio, peak amplitude: {peak:.3f}")
    if peak < 0.02:
        print(">> WARNING: peak amplitude is very low -- check your mic "
              "input device / gain with sd.query_devices() before assuming "
              "Whisper is at fault.")

    return audio


def transcribe(audio: np.ndarray) -> str:
    """Runs Faster-Whisper on recorded audio, returns the transcript text."""
    if audio.size == 0:
        return ""

    model = _get_whisper_model()
    segments, info = model.transcribe(
        audio,
        beam_size=5,
        language="en",
        # Strips silence/noise before transcribing -- without this,
        # Whisper often "hallucinates" words to fill silent gaps at the
        # start/end of a push-to-talk recording, which is the most common
        # cause of "it transcribed something I never said".
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        # Stops one bad guess early in the audio from skewing everything
        # that follows -- default behavior uses prior output as context,
        # which can snowball a single misheard word into a wrong sentence.
        condition_on_previous_text=False,
        # Nudges Whisper toward the right domain/vocabulary.
        initial_prompt=(
            "This is a spoken answer in a technical job interview, likely "
            "covering software engineering, computer science, or personal "
            "background."
        ),
    )
    text = " ".join(segment.text.strip() for segment in segments)
    return text.strip()


def record_and_transcribe() -> str:
    audio = record_audio()
    return transcribe(audio)


# ---------------------------------------------------------------------
# TTS: Kokoro
# ---------------------------------------------------------------------

_kokoro_pipeline = None


def _get_kokoro_pipeline():
    global _kokoro_pipeline
    if _kokoro_pipeline is None:
        from kokoro import KPipeline
        _kokoro_pipeline = KPipeline(lang_code="a")  # 'a' = American English
    return _kokoro_pipeline


def speak(text: str, voice: str = "af_heart") -> None:
    """
    Generates speech for `text` with Kokoro and plays it immediately.
    Kokoro yields audio in chunks; we play each chunk as it's ready
    instead of waiting for the whole sentence, so the interviewer's
    voice starts almost instantly.
    """
    print(f"\n[Interviewer says]: {text}\n")

    pipeline = _get_kokoro_pipeline()
    generator = pipeline(text, voice=voice)

    for _, _, audio_chunk in generator:
        # audio_chunk is a numpy float32 array at 24kHz
        sd.play(audio_chunk, samplerate=24000)
        sd.wait()  # block until this chunk finishes before playing the next

import io
import threading
import numpy as np
import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 16000

_whisper_model = None


def _get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        _whisper_model = WhisperModel(
            "small.en", device="cpu", compute_type="int8"
        )
    return _whisper_model


def record_audio() -> np.ndarray:
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
        input()
        stop_event.set()

    print(">> Recording stopped. Transcribing...")

    if not frames:
        return np.array([], dtype=np.float32)

    audio = np.concatenate(frames, axis=0).flatten()

    duration = len(audio) / SAMPLE_RATE
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    print(f">> Captured {duration:.1f}s of audio, peak amplitude: {peak:.3f}")
    if peak < 0.02:
        print(">> WARNING: peak amplitude is very low -- check your mic "
              "input device / gain with sd.query_devices() before assuming "
              "Whisper is at fault.")

    return audio


def transcribe(audio: np.ndarray) -> str:
    if audio.size == 0:
        return ""

    model = _get_whisper_model()
    segments, info = model.transcribe(
        audio,
        beam_size=5,
        language="en",
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        condition_on_previous_text=False,
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


_kokoro_pipeline = None


def _get_kokoro_pipeline():
    global _kokoro_pipeline
    if _kokoro_pipeline is None:
        from kokoro import KPipeline
        _kokoro_pipeline = KPipeline(lang_code="a")
    return _kokoro_pipeline


def speak(text: str, voice: str = "af_heart") -> None:
    print(f"\n[Interviewer says]: {text}\n")

    pipeline = _get_kokoro_pipeline()
    generator = pipeline(text, voice=voice)

    for _, _, audio_chunk in generator:
        sd.play(audio_chunk, samplerate=24000)
        sd.wait()


def synthesize_speech_bytes(text: str, voice: str = "af_heart") -> bytes:
    pipeline = _get_kokoro_pipeline()
    generator = pipeline(text, voice=voice)

    chunks = [audio_chunk for _, _, audio_chunk in generator]
    if not chunks:
        return b""

    full_audio = np.concatenate(chunks)
    buffer = io.BytesIO()
    sf.write(buffer, full_audio, samplerate=24000, format="WAV")
    buffer.seek(0)
    return buffer.read()


def transcribe_uploaded_audio(audio_bytes: bytes) -> str:
    audio, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sample_rate != SAMPLE_RATE:
        duration = len(audio) / sample_rate
        target_len = int(duration * SAMPLE_RATE)
        audio = np.interp(
            np.linspace(0, len(audio), target_len, endpoint=False),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)

    return transcribe(audio)

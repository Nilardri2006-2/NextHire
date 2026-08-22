"""
main.py
-------
Runs the graph:  START -> load_resume_and_policy -> screening -> END

Since `screening` now handles the whole speak/listen loop for every
question itself (it calls speak() and record_and_transcribe() directly
inside the node), main.py just needs to preload the voice models once
and invoke the graph a single time -- no manual looping here anymore.

Run with:
    python main.py
"""

import time
from graph import interview_graph
from voice_utils import _get_whisper_model, _get_kokoro_pipeline


def preload_models():
    print("Loading models (one-time cost)...")
    t0 = time.time()
    _get_whisper_model()
    print(f"  Whisper loaded in {time.time() - t0:.1f}s")

    t0 = time.time()
    _get_kokoro_pipeline()
    print(f"  Kokoro loaded in {time.time() - t0:.1f}s")

    print("Ready.\n")


if __name__ == "__main__":
    preload_models()

    candidate_name = input("Candidate name: ").strip() or "Candidate"

    result = interview_graph.invoke({
        "candidate_name": candidate_name,
        "question": "",
        "resume_text": "",
        "policy_text": "",
        "ans": "",
        "history": [],
        "llm_call": 0,
    })

    print("\n" + "=" * 60)
    print("Screening round finished. Full transcript:")
    print("=" * 60)
    for i, turn in enumerate(result["history"], 1):
        print(f"\nQ{i}: {turn['question']}")
        print(f"A{i}: {turn['answer']}")

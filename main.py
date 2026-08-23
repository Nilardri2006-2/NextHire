"""
main.py
-------
Runs the graph:  START -> load_resume_and_policy -> screening -> final ->
                  generate_pdf -> END

Now backed by a Postgres checkpointer (via Render's DATABASE_URL), so the
graph's state is saved to the database at every node -- if the process
crashes or you stop it mid-interview, you can resume from the last
completed node instead of starting over.

The Postgres connection has to stay open for as long as the graph is
running, so it's opened here (as a `with` block) rather than at import
time in graph.py -- that's the one structural change from before.

Run with:
    python main.py
"""

import os
import time

from dotenv import load_dotenv
from langgraph.checkpoint.postgres import PostgresSaver

from graph import build_graph
from voice_utils import _get_whisper_model, _get_kokoro_pipeline

load_dotenv()


def get_database_url() -> str:
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError(
            "DATABASE_URL is not set. Add it to your .env file -- copy the "
            "'External Database URL' from your Render Postgres dashboard."
        )

    # Render requires SSL for external connections; make sure it's set.
    if "sslmode=" not in db_url:
        separator = "&" if "?" in db_url else "?"
        db_url = f"{db_url}{separator}sslmode=require"

    return db_url


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
    db_url = get_database_url()

    with PostgresSaver.from_conn_string(db_url) as checkpointer:
        # Only needs to actually create tables the first time -- safe to
        # call on every run after that (it won't touch existing tables).
        checkpointer.setup()

        interview_graph = build_graph(checkpointer=checkpointer)

        # thread_id groups all checkpoints for this one interview session.
        # Use something stable per-candidate-per-run so you could resume
        # this exact session later by invoking with the same thread_id.
        config = {"configurable": {"thread_id": candidate_name}}

        result = interview_graph.invoke({
            "candidate_name": candidate_name,
            "question": "",
            "resume_text": "",
            "policy_text": "",
            "ans": "",
            "history": [],
            "llm_call": 0,
            "evaluation": "",
            "pdf_path": "",
        }, config=config)

    print("\n" + "=" * 60)
    print("Screening round finished. Full transcript:")
    print("=" * 60)

    print("\n history:", result["history"])
    print("\n" + "=" * 60)
    print("Evaluation:")
    print("=" * 60)
    print(result["evaluation"])
    print(f"\nReport saved to: {result['pdf_path']}")
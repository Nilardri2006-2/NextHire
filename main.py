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
        checkpointer.setup()

        interview_graph = build_graph(checkpointer=checkpointer)

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

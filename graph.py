"""
graph.py
--------
Graph so far:

    START -> load_resume_and_policy -> screening -> END

- load_resume_and_policy: reads the resume + hiring-policy PDFs, asks the
  LLM to summarize each (this was the bug before -- the raw extracted
  text wasn't actually being passed into the summarize prompt, so it was
  summarizing nothing). Fixed below.

- screening: runs the full screening round as a live voice conversation.
  Asks the first question from resume_text + policy_text, speaks it,
  listens for and transcribes the answer, stores the Q&A pair in
  history, then keeps asking new (non-repeated) questions grounded in
  the resume, the policy's screening guide, and everything asked so far.
  Prints each question, each transcribed answer, and a "node complete"
  line at the end.

You'll scale this out with more nodes later (technical rounds,
behavioral, etc.) -- this is deliberately just the screening node,
fully working end to end.
"""

import os
from typing import TypedDict, List, Dict

from langchain_community.document_loaders import PyPDFLoader
from langgraph.graph import StateGraph, START, END
from langchain_cohere import ChatCohere
from dotenv import load_dotenv

from voice_utils import speak, record_and_transcribe

load_dotenv()

# ---------------------------------------------------------------------
# File paths -- point these at your actual PDF files.
# ---------------------------------------------------------------------
RESUME_PDF_PATH = "NILARDRI_PRAMANICK_RESUME_v4.pdf"
POLICY_PDF_PATH = "xyz hiring policy.pdf"

# How many questions the screening round asks in total (1st + follow-ups).
NUM_SCREENING_QUESTIONS = 3

llm = ChatCohere(
    cohere_api_key=os.getenv("COHERE_API_KEY"),
    temperature=0.1,
)


# ---------------------------------------------------------------------
# 1. State schema
# ---------------------------------------------------------------------

class InterviewState(TypedDict):
    candidate_name: str
    question: str
    resume_text: str
    policy_text: str
    ans: str
    history: List[Dict]
    llm_call: int


# ---------------------------------------------------------------------
# 2. load_resume_and_policy node
# ---------------------------------------------------------------------

def load_resume_and_policy(state: InterviewState) -> dict:
    resume_docs = PyPDFLoader(RESUME_PDF_PATH).load()
    raw_resume_text = "\n".join(doc.page_content for doc in resume_docs)

    policy_docs = PyPDFLoader(POLICY_PDF_PATH).load()
    raw_policy_text = "\n".join(doc.page_content for doc in policy_docs)

    if raw_resume_text:
        print("resume loaded")
    if raw_policy_text:
        print("policy loaded")

    r_prompt = f"""You are an HR assistant. You are given a candidate's resume text below.
Summarize it in under 100 words, keeping the key points: projects,
experience, and skills.

Resume text:
{raw_resume_text}
"""
    resume_summary = llm.invoke(r_prompt).content.strip()

    p_prompt = f"""You are an HR assistant. You are given a company hiring policy
document below. Summarize it in under 100 words. Include any specific
questions that appear in the policy, and any guidance for how the
screening round should be run.

Policy text:
{raw_policy_text}
"""
    policy_summary = llm.invoke(p_prompt).content.strip()

    print("[node complete] load_resume_and_policy")

    return {
        "resume_text": resume_summary,
        "policy_text": policy_summary,
        "llm_call": state.get("llm_call", 0) + 2,
    }


# ---------------------------------------------------------------------
# 3. screening node
# ---------------------------------------------------------------------

def screening(state: InterviewState) -> dict:
    resume_text = state["resume_text"]
    policy_text = state["policy_text"]
    history = list(state.get("history") or [])

    for i in range(NUM_SCREENING_QUESTIONS):
        if i == 0:
            prompt = f"""You are an HR assistant conducting the screening round of a job
interview. Using the candidate's resume summary and the company's
hiring policy (which includes the required screening questions and
guidance) below, ask your FIRST screening question.

Resume summary:
{resume_text}

Hiring policy / screening question guide:
{policy_text}

Reply with exactly one clear, natural-sounding spoken question. No
preamble, no numbering, no markdown -- just the question itself.
"""
        else:
            history_text = "\n".join(
                f"Q: {h['question']}\nA: {h['answer']}" for h in history
            ) or "(none yet)"

            prompt = f"""You are an HR assistant continuing the screening round of a job
interview.

Resume summary:
{resume_text}

Hiring policy / screening question guide:
{policy_text}

Questions already asked and answered so far:
{history_text}

Ask exactly one NEW screening question that has not been asked yet.
Base it on the resume, the policy's screening guide, and the
candidate's previous answers -- do not repeat a question from history.
Reply with just the question itself, no preamble, no numbering, no
markdown.
"""

        question = llm.invoke(prompt).content.strip()
        print(f"\n[Screening Q{i + 1}]: {question}")

        speak(question)                      # TTS: interviewer asks it out loud
        answer = record_and_transcribe()      # STT: candidate's spoken answer

        print(f"[Candidate's answer]: {answer}\n")

        history.append({"question": question, "answer": answer})

    print("[node complete] screening")

    return {
        "history": history,
        "question": history[-1]["question"] if history else "",
        "llm_call": state.get("llm_call", 0) + NUM_SCREENING_QUESTIONS,
    }


# ---------------------------------------------------------------------
# 4. Wire up the graph: START -> load_resume_and_policy -> screening -> END
# ---------------------------------------------------------------------

def build_graph():
    builder = StateGraph(InterviewState)

    builder.add_node("load_resume_and_policy", load_resume_and_policy)
    builder.add_node("screening", screening)

    builder.add_edge(START, "load_resume_and_policy")
    builder.add_edge("load_resume_and_policy", "screening")
    builder.add_edge("screening", END)

    return builder.compile()


interview_graph = build_graph()


if __name__ == "__main__":
    result = interview_graph.invoke({
        "candidate_name": "Roney",
        "question": "",
        "resume_text": "",
        "policy_text": "",
        "ans": "",
        "history": [],
        "llm_call": 0,
    })
    print("\nFinal history:")
    for turn in result["history"]:
        print(turn)

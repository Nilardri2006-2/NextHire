"""
graph.py
--------
Graph so far:

    START -> load_resume_and_policy -> screening -> END

- load_resume_and_policy: reads the resume + hiring-policy PDFs, summarizes
  the resume, extracts the structured hiring policy (round name, guide,
  questions list), then runs a SEPARATE dedicated LLM call whose only job
  is to count how many questions are in each round -- that count is what
  gets stored in policy.rounds[i].number_of_questions.

- screening: runs the full screening round as a live voice conversation.
  Reads policy.rounds[0].number_of_questions to know how many questions to
  ask, and policy.rounds[0].questions as the actual question list. Asks
  the first question, speaks it, listens for and transcribes the answer,
  stores the Q&A pair in history, then keeps asking new (non-repeated)
  questions grounded in the resume, the policy's screening guide, and
  everything asked so far. Prints each question, each transcribed answer,
  and a "node complete" line at the end.

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

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER



load_dotenv()

# ---------------------------------------------------------------------
# Folder layout:
#   Folder/Input/  -- resume + policy PDFs go here (you place them)
#   Folder/Output/ -- generated evaluation report PDF goes here
# ---------------------------------------------------------------------
INPUT_DIR = os.path.join("Folder", "Input")
OUTPUT_DIR = os.path.join("Folder", "Output")

RESUME_PDF_PATH = os.path.join(INPUT_DIR, "NILARDRI_PRAMANICK_RESUME_v4.pdf")
POLICY_PDF_PATH = os.path.join(INPUT_DIR, "xyz hiring policy_3.pdf")

llm = ChatCohere(
    cohere_api_key=os.getenv("COHERE_API_KEY"),
    temperature=0.1,
)

# ---------------------------------------------------------------------
# 0. Models
# ---------------------------------------------------------------------
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class InterviewRound(BaseModel):
    round_name: str = Field(
        description="Name of the interview round"
    )

    guide: str = Field(
        description="Instructions and guidelines for conducting this round"
    )

    questions: List[str] = Field(
        description="List of questions specified in the policy for this round"
    )

    number_of_questions: int = Field(
        description="Total number of questions for this round"
    )


class HiringPolicy(BaseModel):
    rounds: List[InterviewRound] = Field(
        description="All interview rounds defined in the hiring policy"
    )


class QuestionCount(BaseModel):
    count: int = Field(
        description="Total number of distinct questions listed for this interview round"
    )


# ---------------------------------------------------------------------
# 1. State schema
# ---------------------------------------------------------------------

class InterviewState(TypedDict):
    candidate_name: str
    question: str
    resume_text: str
    policy_text: str
    policy: HiringPolicy
    ans: str
    history: List[Dict]
    llm_call: int
    evaluation: str
    pdf_path: str


# ---------------------------------------------------------------------
# 2. load_resume_and_policy node
# ---------------------------------------------------------------------

def load_resume_and_policy(state: InterviewState) -> dict:
    if not os.path.isfile(RESUME_PDF_PATH):
        raise FileNotFoundError(
            f"Resume PDF not found at '{RESUME_PDF_PATH}'. "
            f"Place it inside the '{INPUT_DIR}' folder."
        )
    if not os.path.isfile(POLICY_PDF_PATH):
        raise FileNotFoundError(
            f"Policy PDF not found at '{POLICY_PDF_PATH}'. "
            f"Place it inside the '{INPUT_DIR}' folder."
        )

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

    p_prompt = f"""You are an HR assistant. Read the hiring policy below and convert it into a structured
interview plan.

For EVERY interview round mentioned in the policy:

1. Extract the round name.
2. Extract the guide/instructions for conducting that round.
3. Extract the questions explicitly mentioned in the policy.
4. Determine the total number of questions for that round.

Policy text:
{raw_policy_text}
"""
    structured_policy_llm = llm.with_structured_output(HiringPolicy)
    policy = structured_policy_llm.invoke(p_prompt)

    # Dedicated counting pass: a SEPARATE LLM call whose only job is to
    # count how many questions belong to each round. Keeping this as its
    # own focused call (instead of trusting the count that came out of
    # the single big structured-extraction call above) makes the count
    # more reliable -- a model doing one narrow task at a time tends to
    # get it right more consistently than a model juggling round_name +
    # guide + questions + count all in one pass.
    count_llm = llm.with_structured_output(QuestionCount)
    for interview_round in policy.rounds:
        count_prompt = f"""Count exactly how many separate, distinct questions are listed for
this interview round. Count each question once, even if the guide text
also restates or references it.

Round name: {interview_round.round_name}

Guide:
{interview_round.guide}

Questions list:
{interview_round.questions}

Return only the count as an integer.
"""
        count_result = count_llm.invoke(count_prompt)
        interview_round.number_of_questions = count_result.count

    print("[node complete] load_resume_and_policy")

    return {
        "resume_text": resume_summary,
        "policy": policy,
        "llm_call": state.get("llm_call", 0) + 2 + len(policy.rounds),
    }


# ---------------------------------------------------------------------
# 3. screening node
# ---------------------------------------------------------------------

def screening(state: InterviewState) -> dict:
    resume_text = state["resume_text"]
    policy = state["policy"].rounds[0]  # assuming its screening node
    num_questions = policy.number_of_questions
    history = list(state.get("history") or [])

    for i in range(num_questions):
        if i == 0:
            prompt = f"""You are an HR assistant conducting the screening round of a job
interview. Using the candidate's resume summary and the company's
hiring policy (which includes the required screening questions and
guidance) below, ask your FIRST screening question.

Resume summary:
{resume_text}

Hiring policy / screening question guide:
{policy.guide}

Required questions from the policy, in order:
{policy.questions}

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
{policy.guide}

The questions:
{policy.questions}

Questions already asked and answered so far:
{history_text}
Ask exactly one NEW screening question that has not been asked yet.
Base it on the resume and STRICTLY follw the policy's screening guide, and the
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
        "llm_call": state.get("llm_call", 0) + num_questions,
    }


# ---------------------------------------------------------------------
# 4. Final Agent
# ---------------------------------------------------------------------
def final(state: InterviewState) -> dict:
    resume_text = state["resume_text"]
    policy = state["policy"]
    history = state.get("history")
    prompt = f"""
You are an HR evaluator assessing a candidate for the role.
Evaluate based on:
Resume:
{resume_text}
Screening Guidelines:
{policy}
Interview Q&A:
{history}
Evaluate pointwise:

1. Communication
2. Technical Understanding
3. Problem Solving
4. Resume Credibility
5. Role Fit
6. Motivation
7. Professionalism
8. Strengths
9. Weaknesses
10. Final Recommendation
For each scored criterion, give a score out of 10 and one brief justification.
Use only evidence from the resume and interview.
"""
    evaluation = llm.invoke(prompt).content.strip()
    print("evaluation done")

    return {
        "evaluation": evaluation,
        "llm_call": state.get("llm_call",0)+1
    }

# ---------------------------------------------------------------------
# 5. pdf generation
# ---------------------------------------------------------------------

def generate_pdf(state: InterviewState) -> dict:

    candidate_name = state["candidate_name"]
    resume_text = state["resume_text"]
    policy = state["policy"]
    history = state.get("history") or []
    evaluation = state["evaluation"]

    # Output PDF goes in Folder/Output/
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # PDF filename
    safe_name = candidate_name.replace(" ", "_")

    pdf_path = os.path.join(
        OUTPUT_DIR,
        f"{safe_name}_interview_report.pdf"
    )

    # PDF setup
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        rightMargin=50,
        leftMargin=50,
        topMargin=50,
        bottomMargin=50,
    )
    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    title_style.alignment = TA_CENTER
    heading_style = styles["Heading2"]
    body_style = styles["BodyText"]
    story = []

    # Title
    story.append(
        Paragraph(
            "Candidate Interview Evaluation Report",
            title_style
        )
    )
    story.append(Spacer(1, 20))
    story.append(
        Paragraph(
            f"<b>Candidate:</b> {candidate_name}",
            body_style
        )
    )
    story.append(Spacer(1, 20))

    # Resume
    story.append(
        Paragraph(
            "1. Resume Summary",
            heading_style
        )
    )
    story.append(
        Paragraph(
            resume_text.replace("\n", "<br/>"),
            body_style
        )
    )
    story.append(Spacer(1, 20))

    # Policy
    story.append(
        Paragraph(
            "2. Interview Policy",
            heading_style
        )
    )
    for round_info in policy.rounds:
        story.append(
            Paragraph(
                f"<b>Round:</b> {round_info.round_name}",
                body_style
            )
        )
        story.append(
            Paragraph(
                f"<b>Guide:</b> {round_info.guide}",
                body_style
            )
        )
        story.append(
            Paragraph(
                f"<b>Number of Questions:</b> "
                f"{round_info.number_of_questions}",
                body_style
            )
        )
        story.append(Spacer(1, 10))
    story.append(PageBreak())

    # Interview History
    story.append(
        Paragraph(
            "3. Interview Transcript",
            heading_style
        )
    )
    for i, turn in enumerate(history, 1):
        question = turn.get("question", "")
        answer = turn.get("answer", "")
        story.append(
            Paragraph(
                f"<b>Question {i}:</b> {question}",
                body_style
            )
        )
        story.append(Spacer(1, 5))
        story.append(
            Paragraph(
                f"<b>Answer {i}:</b> {answer}",
                body_style
            )
        )
        story.append(Spacer(1, 15))
    story.append(PageBreak())

    # Evaluation
    story.append(
        Paragraph(
            "4. Candidate Evaluation",
            heading_style
        )
    )
    story.append(
        Paragraph(
            evaluation.replace("\n", "<br/>"),
            body_style
        )
    )

    # Generate PDF
    doc.build(story)
    print(f"\nPDF generated successfully:")
    print(pdf_path)

    return {"pdf_path": pdf_path}
# ---------------------------------------------------------------------
# 6. Wire up the graph: START -> load_resume_and_policy -> screening -> END
# ---------------------------------------------------------------------

def build_graph():
    builder = StateGraph(InterviewState)

    builder.add_node("load_resume_and_policy", load_resume_and_policy)
    builder.add_node("screening", screening)
    builder.add_node("final", final)
    builder.add_node("generate_pdf", generate_pdf)

    builder.add_edge(START, "load_resume_and_policy")
    builder.add_edge("load_resume_and_policy", "screening")
    builder.add_edge("screening", "final")
    builder.add_edge("final", "generate_pdf")
    builder.add_edge("generate_pdf" , END)

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
        "evaluation": "",
        "pdf_path": "",
    })
    print("\nFinal history:")
    for turn in result["history"]:
        print(turn)
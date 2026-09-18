import base64
import os
import uuid

from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel

from graph import load_resume_and_policy, final, generate_pdf, INPUT_DIR
from api_helpers import generate_question
from voice_utils import synthesize_speech_bytes, transcribe_uploaded_audio

app = FastAPI(title="NextHire Interview API")

SESSIONS: dict = {}


class StartResponse(BaseModel):
    session_id: str
    question: str
    audio_base64: str
    done: bool = False


class AnswerResponse(BaseModel):
    done: bool
    question: str | None = None
    audio_base64: str | None = None
    evaluation: str | None = None
    pdf_path: str | None = None


@app.post("/start", response_model=StartResponse)
async def start_interview(
    candidate_name: str = Form(...),
    resume: UploadFile = File(...),
    policy: UploadFile = File(...),
):
    session_id = str(uuid.uuid4())

    os.makedirs(INPUT_DIR, exist_ok=True)
    resume_path = os.path.join(INPUT_DIR, f"{session_id}_resume.pdf")
    policy_path = os.path.join(INPUT_DIR, f"{session_id}_policy.pdf")

    with open(resume_path, "wb") as f:
        f.write(await resume.read())
    with open(policy_path, "wb") as f:
        f.write(await policy.read())

    state = {
        "candidate_name": candidate_name,
        "history": [],
        "resume_pdf_path": resume_path,
        "policy_pdf_path": policy_path,
    }

    state.update(load_resume_and_policy(state))

    policy_round = state["policy"].rounds[0]
    question = generate_question(state["resume_text"], policy_round, [])
    audio_bytes = synthesize_speech_bytes(question)

    SESSIONS[session_id] = {
        "candidate_name": candidate_name,
        "resume_text": state["resume_text"],
        "policy": state["policy"],
        "policy_round": policy_round,
        "history": [],
        "current_question": question,
        "num_questions": policy_round.number_of_questions,
        "asked": 1,
    }

    return StartResponse(
        session_id=session_id,
        question=question,
        audio_base64=base64.b64encode(audio_bytes).decode(),
    )


@app.post("/answer", response_model=AnswerResponse)
async def submit_answer(
    session_id: str = Form(...),
    audio: UploadFile = File(...),
):
    session = SESSIONS[session_id]

    audio_bytes = await audio.read()
    answer_text = transcribe_uploaded_audio(audio_bytes)

    session["history"].append({
        "question": session["current_question"],
        "answer": answer_text,
    })

    if session["asked"] < session["num_questions"]:
        next_question = generate_question(
            session["resume_text"], session["policy_round"], session["history"]
        )
        session["current_question"] = next_question
        session["asked"] += 1

        audio_out = synthesize_speech_bytes(next_question)
        return AnswerResponse(
            done=False,
            question=next_question,
            audio_base64=base64.b64encode(audio_out).decode(),
        )

    full_state = {
        "candidate_name": session["candidate_name"],
        "resume_text": session["resume_text"],
        "policy": session["policy"],
        "history": session["history"],
    }
    full_state.update(final(full_state))
    full_state.update(generate_pdf(full_state))

    return AnswerResponse(
        done=True,
        evaluation=full_state["evaluation"],
        pdf_path=full_state["pdf_path"],
    )

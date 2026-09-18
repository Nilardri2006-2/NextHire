from graph import llm


def build_first_question_prompt(resume_text: str, policy_round) -> str:
    return f"""You are an HR assistant conducting the screening round of a job
interview. Using the candidate's resume summary and the company's
hiring policy (which includes the required screening questions and
guidance) below, ask your FIRST screening question.

Resume summary:
{resume_text}

Hiring policy / screening question guide:
{policy_round.guide}

Required questions from the policy, in order:
{policy_round.questions}

Reply with exactly one clear, natural-sounding spoken question. No
preamble, no numbering, no markdown -- just the question itself.
"""


def build_followup_question_prompt(resume_text: str, policy_round, history: list) -> str:
    history_text = "\n".join(
        f"Q: {h['question']}\nA: {h['answer']}" for h in history
    ) or "(none yet)"

    return f"""You are an HR assistant continuing the screening round of a job
interview.

Resume summary:
{resume_text}

Hiring policy / screening question guide:
{policy_round.guide}

The questions:
{policy_round.questions}

Questions already asked and answered so far:
{history_text}
Ask exactly one NEW screening question that has not been asked yet.
Base it on the resume and STRICTLY follw the policy's screening guide, and the
candidate's previous answers -- do not repeat a question from history.
Reply with just the question itself, no preamble, no numbering, no
markdown.
"""


def generate_question(resume_text: str, policy_round, history: list) -> str:
    if not history:
        prompt = build_first_question_prompt(resume_text, policy_round)
    else:
        prompt = build_followup_question_prompt(resume_text, policy_round, history)

    return llm.invoke(prompt).content.strip()

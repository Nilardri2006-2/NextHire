import base64
import requests
import streamlit as st

BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="NextHire Interview", page_icon="🎙️", layout="centered")

if "session_id" not in st.session_state:
    st.session_state.session_id = None
    st.session_state.question = ""
    st.session_state.audio_b64 = None
    st.session_state.phase = "idle"
    st.session_state.pdf_path = None

st.markdown(
    "<h2 style='text-align: center;'>NextHire — Practice Interview</h2>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align: center; color: gray;'>"
    "Upload your resume and a question guide to practice a screening "
    "interview at your own pace.</p>",
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("interviewer.png", use_container_width=True)

    status_placeholder = st.empty()
    if st.session_state.phase == "speaking":
        status_placeholder.markdown(
            "<p style='text-align:center; font-size:20px;'>🗣️ <b>Speaking...</b></p>",
            unsafe_allow_html=True,
        )
    elif st.session_state.phase == "listening":
        status_placeholder.markdown(
            "<p style='text-align:center; font-size:20px;'>🎙️ <b>Listening...</b></p>",
            unsafe_allow_html=True,
        )
    elif st.session_state.phase == "finished":
        status_placeholder.markdown(
            "<p style='text-align:center; font-size:20px;'>✅ <b>Interview complete</b></p>",
            unsafe_allow_html=True,
        )

st.divider()

if st.session_state.phase == "idle":
    candidate_name = st.text_input("Your name")
    resume_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
    policy_file = st.file_uploader(
        "Upload the interview question guide (PDF)", type=["pdf"],
        help="A PDF listing the questions you want to be asked, and any "
             "guidance on how the round should be run.",
    )

    ready = bool(candidate_name and resume_file and policy_file)

    if st.button("Start Interview", type="primary", disabled=not ready):
        with st.spinner("Reading your resume and question guide..."):
            files = {
                "resume": (resume_file.name, resume_file.getvalue(), "application/pdf"),
                "policy": (policy_file.name, policy_file.getvalue(), "application/pdf"),
            }
            response = requests.post(
                f"{BACKEND_URL}/start",
                data={"candidate_name": candidate_name},
                files=files,
            )
            response.raise_for_status()
            data = response.json()

        st.session_state.session_id = data["session_id"]
        st.session_state.question = data["question"]
        st.session_state.audio_b64 = data["audio_base64"]
        st.session_state.phase = "speaking"
        st.rerun()

elif st.session_state.phase == "speaking":
    audio_bytes = base64.b64decode(st.session_state.audio_b64)
    st.audio(audio_bytes, format="audio/wav", autoplay=True)

    if st.button("I'm ready to answer"):
        st.session_state.phase = "listening"
        st.rerun()

elif st.session_state.phase == "listening":
    recording = st.audio_input("Record your answer")

    if recording is not None:
        with st.spinner("Submitting your answer..."):
            files = {"audio": ("answer.wav", recording.getvalue(), "audio/wav")}
            data = {"session_id": st.session_state.session_id}
            response = requests.post(f"{BACKEND_URL}/answer", data=data, files=files)
            response.raise_for_status()
            result = response.json()

        if result["done"]:
            st.session_state.phase = "finished"
            st.session_state.pdf_path = result.get("pdf_path")
        else:
            st.session_state.question = result["question"]
            st.session_state.audio_b64 = result["audio_base64"]
            st.session_state.phase = "speaking"

        st.rerun()

elif st.session_state.phase == "finished":
    st.success("Thank you for completing the screening interview!")

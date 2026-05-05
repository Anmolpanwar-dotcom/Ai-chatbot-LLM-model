import hashlib
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import streamlit as st
from dotenv import load_dotenv

from llm import build_chat_chain
from retriever import build_vectorstore_from_uploads


load_dotenv()

APP_TITLE = "NovaDesk AI"


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "processed_signature" not in st.session_state:
        st.session_state.processed_signature = None
    if "docs_status" not in st.session_state:
        st.session_state.docs_status = None
    if "hf_token" not in st.session_state:
        st.session_state.hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
    if "hf_model" not in st.session_state:
        st.session_state.hf_model = os.getenv("HF_CHAT_MODEL", "")
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.3
    if "max_new_tokens" not in st.session_state:
        st.session_state.max_new_tokens = 512


def build_upload_signature(uploaded_files):
    if not uploaded_files:
        return None
    joined = "||".join(
        f"{file.name}:{len(file.getvalue())}" for file in uploaded_files
    )
    return hashlib.md5(joined.encode("utf-8")).hexdigest()


def apply_styles():
    st.markdown(
        """
        <style>
            html, body, [class*="css"] {
                font-family: 'Segoe UI', ui-sans-serif, system-ui, sans-serif;
            }

            #MainMenu, footer, header,
            [data-testid="stDecoration"],
            [data-testid="stToolbar"] { display: none !important; }

            .block-container { padding: 0 !important; max-width: 100% !important; }

            .stApp { background: #f3f3f3; }

            /* sidebar */
            [data-testid="stSidebar"] {
                background: #f9f9f9 !important;
                border-right: 1px solid #e0e0e0 !important;
            }
            [data-testid="stSidebar"] > div:first-child { padding: 14px 12px; }
            [data-testid="stSidebar"] label,
            [data-testid="stSidebar"] .stMarkdown p,
            [data-testid="stSidebar"] span { color: #444 !important; font-size: 13px; }
            [data-testid="stSidebar"] .stButton > button {
                background: transparent !important;
                border: 1px solid #d0d0d0 !important;
                color: #222 !important;
                border-radius: 8px !important;
                font-size: 13px !important;
                width: 100% !important;
            }
            [data-testid="stSidebar"] .stButton > button:hover { background: #efefef !important; }
            [data-testid="stExpander"] {
                background: #fff !important;
                border: 1px solid #e0e0e0 !important;
                border-radius: 10px !important;
            }
            [data-testid="stExpander"] summary,
            [data-testid="stExpander"] label,
            [data-testid="stExpander"] p,
            [data-testid="stExpander"] span { color: #333 !important; font-size: 13px !important; }
            [data-testid="stExpander"] input,
            [data-testid="stExpander"] textarea {
                background: #f9f9f9 !important;
                color: #111 !important;
                border: 1px solid #ddd !important;
                border-radius: 8px !important;
                font-size: 13px !important;
            }
            [data-testid="stFileUploader"] {
                background: #fff !important;
                border: 1px dashed #ccc !important;
                border-radius: 10px !important;
            }
            [data-testid="stFileUploader"] * { color: #888 !important; font-size: 13px !important; }

            /* welcome screen */
            .welcome-wrap {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                min-height: 72vh;
            }
            .welcome-title {
                font-size: 32px;
                font-weight: 400;
                color: #111;
                margin-bottom: 32px;
                text-align: center;
                letter-spacing: -0.3px;
            }
            .pill-row {
                display: flex;
                gap: 10px;
                justify-content: center;
                flex-wrap: wrap;
            }
            .pill {
                display: inline-flex;
                align-items: center;
                gap: 8px;
                background: #fff;
                border: 1px solid #d8d8d8;
                border-radius: 999px;
                padding: 10px 20px;
                font-size: 13.5px;
                color: #333;
                white-space: nowrap;
            }
            .pill svg { width: 15px; height: 15px; flex-shrink: 0; }

            /* messages */
            .stChatMessage {
                background: transparent !important;
                border: none !important;
                padding: 8px 0 !important;
                max-width: 720px;
                margin: 0 auto;
            }
            .stChatMessage p, .stChatMessage li,
            .stChatMessage span, .stChatMessage div {
                color: #111 !important;
                font-size: 15px !important;
                line-height: 1.7 !important;
            }
            [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) .stChatMessageContent {
                background: #fff !important;
                border: 1px solid #e0e0e0 !important;
                border-radius: 18px 18px 4px 18px !important;
                padding: 10px 16px !important;
                display: inline-block !important;
                max-width: 78% !important;
                float: right !important;
            }
            [data-testid="chatAvatarIcon-assistant"] {
                background: #111 !important; color: #fff !important; border-radius: 50% !important;
            }
            [data-testid="chatAvatarIcon-user"] {
                background: #5436DA !important; color: #fff !important; border-radius: 50% !important;
            }

            /* input bar */
            [data-testid="stChatInputContainer"] {
                background: #fff !important;
                border: 1px solid #d0d0d0 !important;
                border-radius: 999px !important;
                padding: 6px 14px !important;
                max-width: 720px;
                margin: 0 auto;
            }
            [data-testid="stChatInputContainer"]:focus-within { border-color: #aaa !important; }
            [data-testid="stChatInput"] {
                background: transparent !important;
                color: #111 !important;
                font-size: 15px !important;
            }
            [data-testid="stChatInput"]::placeholder { color: #aaa !important; }
            [data-testid="stChatInputSubmitButton"] > button {
                background: #111 !important;
                border-radius: 50% !important;
                border: none !important;
                width: 34px !important;
                height: 34px !important;
            }
            [data-testid="stChatInputSubmitButton"] > button:hover { background: #333 !important; }

            .doc-pill {
                display: inline-flex; align-items: center; gap: 6px;
                background: #fff; border: 1px solid #ddd;
                border-radius: 8px; padding: 5px 12px;
                font-size: 12px; color: #666;
            }
            .doc-pill .dot { width: 7px; height: 7px; border-radius: 50%; background: #22c55e; }
            .hint { font-size: 11px; color: #bbb; text-align: center; margin-top: 8px; }
            hr { display: none !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    with st.sidebar:
        st.markdown(
            """
            <div style="display:flex;align-items:center;gap:10px;padding:6px 4px 18px;">
              <div style="width:30px;height:30px;border-radius:50%;background:#111;
                display:flex;align-items:center;justify-content:center;
                font-size:13px;font-weight:700;color:#fff;">N</div>
              <span style="font-size:15px;font-weight:600;color:#111;">NovaDesk AI</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("＋  New chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.vectorstore = None
            st.session_state.processed_signature = None
            st.session_state.docs_status = None
            st.rerun()

        st.markdown(
            '<div style="font-size:11px;color:#aaa;padding:16px 4px 6px;letter-spacing:0.05em;text-transform:uppercase;">Settings</div>',
            unsafe_allow_html=True,
        )

        with st.expander("Model Settings", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.hf_token = st.text_input(
                    "HF Token", value=st.session_state.hf_token, type="password"
                )
                st.session_state.temperature = st.slider(
                    "Temperature", 0.0, 1.0, st.session_state.temperature, 0.1
                )
            with col2:
                st.session_state.hf_model = st.text_input(
                    "Chat Model", value=st.session_state.hf_model
                )
                st.session_state.max_new_tokens = st.slider(
                    "Max Tokens", 128, 1024, st.session_state.max_new_tokens, 64
                )

        st.markdown(
            '<div style="font-size:11px;color:#aaa;padding:16px 4px 6px;letter-spacing:0.05em;text-transform:uppercase;">Document</div>',
            unsafe_allow_html=True,
        )

        uploaded_files = st.file_uploader(
            "Attach PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="pdf_uploader",
        )

        if st.session_state.docs_status:
            st.markdown(
                f'<div class="doc-pill"><span class="dot"></span>{st.session_state.docs_status}</div>',
                unsafe_allow_html=True,
            )

        return uploaded_files


def auto_process_documents(uploaded_files):
    signature = build_upload_signature(uploaded_files)

    if signature is None:
        if st.session_state.processed_signature is not None:
            st.session_state.vectorstore = None
            st.session_state.processed_signature = None
            st.session_state.docs_status = None
        return

    if signature == st.session_state.processed_signature:
        return

    try:
        with st.spinner("Processing PDF..."):
            st.session_state.vectorstore = build_vectorstore_from_uploads(uploaded_files)
        st.session_state.processed_signature = signature
        st.session_state.docs_status = f"{len(uploaded_files)} PDF ready"
        st.toast("PDF processed", icon="✅")
    except Exception as exc:
        st.session_state.vectorstore = None
        st.session_state.processed_signature = None
        st.session_state.docs_status = None
        st.error(f"PDF processing failed: {exc}")


def render_messages():
    if not st.session_state.messages:
        st.markdown(
            """
            <div class="welcome-wrap">
              <div class="welcome-title">What's on the agenda today?</div>
              <div class="pill-row">
                <div class="pill">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
                    <rect x="3" y="3" width="18" height="18" rx="3"/>
                    <path d="M8 12h8M8 8h5M8 16h6"/>
                  </svg>
                  Summarise a PDF
                </div>
                <div class="pill">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
                    <path d="M12 20h9M16.5 3.5a2.121 2.121 0 013 3L7 19l-4 1 1-4L16.5 3.5z"/>
                  </svg>
                  Write or edit
                </div>
                <div class="pill">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
                    <circle cx="12" cy="12" r="9"/>
                    <path d="M12 8v4l3 3"/>
                  </svg>
                  Look something up
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def generate_reply(user_prompt):
    if not st.session_state.hf_token:
        return "Please add your Hugging Face token in **Model Settings** (sidebar) first."

    try:
        with st.spinner("Thinking..."):
            chain = build_chat_chain(
                hf_token=st.session_state.hf_token,
                model_name=st.session_state.hf_model,
                temperature=st.session_state.temperature,
                max_new_tokens=st.session_state.max_new_tokens,
                vectorstore=st.session_state.vectorstore,
            )
            return chain.invoke(
                {
                    "input": user_prompt,
                    "chat_history": st.session_state.messages[:-1],
                }
            )["answer"]
    except Exception as exc:
        return f"Model request failed: {exc}"


def main():
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="⬛",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_session_state()
    apply_styles()

    uploaded_files = render_sidebar()
    auto_process_documents(uploaded_files)

    render_messages()

    user_prompt = st.chat_input("Ask anything...")

    st.markdown(
        '<div class="hint">NovaDesk AI can make mistakes. Verify important info.</div>',
        unsafe_allow_html=True,
    )

    if user_prompt and user_prompt.strip():
        st.session_state.messages.append({"role": "user", "content": user_prompt.strip()})
        reply = generate_reply(user_prompt.strip())
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()


if __name__ == "__main__":
    main()

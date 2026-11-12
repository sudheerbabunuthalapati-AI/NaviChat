import json
from pathlib import Path
from uuid import uuid4

import streamlit as st
from openai import OpenAI, OpenAIError
from streamlit.errors import StreamlitSecretNotFoundError


APP_DIR = Path(__file__).resolve().parent
CHATS_DIR = APP_DIR / "chats"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
MODEL = "openai/gpt-4o-mini"


def load_conversations() -> dict[str, list[dict[str, str]]]:
    CHATS_DIR.mkdir(parents=True, exist_ok=True)
    conversations = {}

    for chat_file in sorted(
        CHATS_DIR.glob("*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    ):
        try:
            messages = json.loads(chat_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            st.error(f"Could not load {chat_file.name}: {error}")
            continue

        valid_messages = (
            isinstance(messages, list)
            and all(
                isinstance(message, dict)
                and message.get("role") in ("user", "assistant")
                and isinstance(message.get("content"), str)
                for message in messages
            )
        )
        if not valid_messages:
            st.error(f"Could not load {chat_file.name}: invalid chat format.")
            continue

        conversations[chat_file.stem] = messages

    return conversations


def save_conversation(chat_id: str, messages: list[dict[str, str]]) -> bool:
    try:
        CHATS_DIR.mkdir(parents=True, exist_ok=True)
        chat_path = CHATS_DIR / f"{chat_id}.json"
        chat_path.write_text(
            json.dumps(messages, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as error:
        st.error(f"Could not save this chat: {error}")
        return False
    return True


def create_conversation() -> str:
    chat_id = str(uuid4())
    st.session_state.conversations[chat_id] = []
    save_conversation(chat_id, [])
    return chat_id


def chat_label(messages: list[dict[str, str]]) -> str:
    for message in messages:
        if message["role"] == "user":
            text = message["content"].replace("\n", " ").strip()
            return text[:40] + ("..." if len(text) > 40 else "")
    return "New chat"


def chat_options() -> dict[str, str]:
    options = {}
    for chat_id, messages in st.session_state.conversations.items():
        label = chat_label(messages)
        if label in options:
            label = f"{label} ({chat_id[:8]})"
        options[label] = chat_id
    return options


def get_api_key() -> str:
    try:
        return str(st.secrets["OPENROUTER_API_KEY"]).strip()
    except (KeyError, StreamlitSecretNotFoundError):
        return ""


st.set_page_config(page_title="NaviChat", page_icon="💬")
st.title("NaviChat")
st.caption("A beginner-friendly chat app powered by OpenRouter.")

if "conversations" not in st.session_state:
    st.session_state.conversations = load_conversations()
if "active_chat_id" not in st.session_state:
    if st.session_state.conversations:
        st.session_state.active_chat_id = next(iter(st.session_state.conversations))
    else:
        st.session_state.active_chat_id = create_conversation()
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT
if "prompt_draft" not in st.session_state:
    st.session_state.prompt_draft = st.session_state.system_prompt

with st.sidebar:
    st.header("Chats")
    if st.button("＋ New chat", use_container_width=True):
        new_chat_id = create_conversation()
        st.session_state.active_chat_id = new_chat_id
        new_label = next(
            label for label, chat_id in chat_options().items() if chat_id == new_chat_id
        )
        st.session_state.chat_selector = new_label
        st.rerun()

    options = chat_options()
    labels = list(options)
    current_label = next(
        label
        for label, chat_id in options.items()
        if chat_id == st.session_state.active_chat_id
    )
    if st.session_state.get("chat_selector") not in options:
        st.session_state.chat_selector = current_label
    selected_chat_label = st.radio(
        "Your conversations",
        options=labels,
        index=None,
        label_visibility="collapsed",
        key="chat_selector",
    )
    selected_chat_id = options[selected_chat_label]
    st.session_state.active_chat_id = selected_chat_id

    if st.button("Delete current chat", use_container_width=True):
        chat_path = CHATS_DIR / f"{selected_chat_id}.json"
        try:
            chat_path.unlink(missing_ok=True)
        except OSError as error:
            st.error(f"Could not delete this chat: {error}")
        else:
            del st.session_state.conversations[selected_chat_id]
            if st.session_state.conversations:
                next_chat_id = next(iter(st.session_state.conversations))
            else:
                next_chat_id = create_conversation()
            st.session_state.active_chat_id = next_chat_id
            st.rerun()

    st.divider()
    st.header("Settings")
    temperature = st.slider(
        "Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.1
    )
    max_tokens = st.slider(
        "Maximum output tokens", min_value=64, max_value=4096, value=1024, step=64
    )
    context_length = st.slider(
        "Recent messages to include", min_value=1, max_value=30, value=10
    )
    streaming = st.toggle("Stream responses", value=True)

    st.subheader("System prompt")
    st.text_area("Edit the prompt, then save it", key="prompt_draft", height=120)
    if st.button("Save system prompt"):
        st.session_state.system_prompt = st.session_state.prompt_draft
        st.success("System prompt saved for this session.")

    st.caption(
        "Chats are stored as local JSON files. Everyone using one deployment "
        "can see and delete its chats, and Streamlit Cloud may clear local files."
    )

active_messages = st.session_state.conversations[selected_chat_id]
for message in active_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Message NaviChat")
if prompt:
    active_messages.append({"role": "user", "content": prompt})
    save_conversation(selected_chat_id, active_messages)

    with st.chat_message("user"):
        st.markdown(prompt)

    api_key = get_api_key()
    if not api_key or api_key == "paste-your-key-here":
        st.error(
            "Add your OpenRouter key as OPENROUTER_API_KEY in "
            ".streamlit/secrets.toml before sending a message."
        )
    else:
        request_messages = [
            {"role": "system", "content": st.session_state.system_prompt},
            *active_messages[-context_length:],
        ]
        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )

        try:
            with st.chat_message("assistant"):
                if streaming:
                    response_stream = client.chat.completions.create(
                        model=MODEL,
                        messages=request_messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=True,
                    )

                    def response_text():
                        for chunk in response_stream:
                            if chunk.choices:
                                text = chunk.choices[0].delta.content
                                if isinstance(text, str):
                                    yield text

                    assistant_reply = st.write_stream(response_text())
                else:
                    response = client.chat.completions.create(
                        model=MODEL,
                        messages=request_messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    assistant_reply = response.choices[0].message.content or ""
                    st.markdown(assistant_reply)
        except OpenAIError as error:
            st.error(
                f"OpenRouter request failed ({type(error).__name__}). "
                "Check your key, model access, network connection, and usage limits."
            )
        else:
            active_messages.append({"role": "assistant", "content": assistant_reply})
            save_conversation(selected_chat_id, active_messages)

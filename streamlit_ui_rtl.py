from __future__ import annotations
from typing import Literal, TypedDict
import asyncio
import os
import streamlit as st

st.set_page_config(
    page_title="מדריך למשתמש טופס 101",
)

from supabase import Client
from openai import AsyncOpenAI
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    RetryPromptPart,
    ModelMessagesTypeAdapter
)
from ai_agent import ai_agent, AIDeps
from dotenv import load_dotenv

load_dotenv()

# Initialize clients
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = Client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

def get_last_ai_responses(messages: list, count: int = 3) -> list:
    """
    Get the last 'count' AI responses from the message history.
    """
    ai_responses = [
        msg for msg in messages
        if isinstance(msg, ModelResponse) and
        any(isinstance(part, TextPart) for part in msg.parts)
    ]
    return ai_responses[-count:] if ai_responses else []

st.markdown(
    """
    <style>
        .stApp {
            direction: rtl;
        }
        .main .block-container {
            direction: rtl;
            text-align: right;
        }
        .stChatMessage {
            direction: rtl;
            text-align: right;
        }
        .stChatInputContainer {
            direction: rtl;
        }
        p, h1, h2, h3 {
            direction: rtl;
            text-align: right;
        }
        div[data-testid="stMarkdownContainer"] {
            direction: rtl;
            text-align: right;
        }
    </style>
    """,
    unsafe_allow_html=True
)

def display_message_part(part):
    """
    Display a single part of a message in the Streamlit UI.
    """
    if part.part_kind == 'system-prompt':
        with st.chat_message("system"):
            st.markdown(f"**System**: {part.content}")
    elif part.part_kind == 'user-prompt':
        with st.chat_message("user"):
            st.markdown(part.content)
    elif part.part_kind == 'text':
        with st.chat_message("assistant"):
            st.markdown(part.content)

async def run_agent_with_streaming(user_input: str, model_context: list):
    """
    Run the agent with streaming text support for RTL.
    'model_context' now includes the last 3 AI responses plus the current user input.
    """
    deps = AIDeps(
        supabase=supabase,
        openai_client=openai_client
    )

    async with ai_agent.run_stream(
        user_input,
        deps=deps,
        message_history=model_context,
    ) as result:
        partial_text = ""
        message_placeholder = st.empty()

        async for chunk in result.stream_text(delta=True):
            partial_text += chunk
            message_placeholder.markdown(
                f"<div dir='rtl'>{partial_text}</div>",
                unsafe_allow_html=True
            )

        # Filter and clean any new assistant messages
        filtered_messages = [
            msg for msg in result.new_messages()
            if not (
                hasattr(msg, 'parts') and
                any(part.part_kind == 'user-prompt' for part in msg.parts)
            )
        ]

        # Add the new assistant messages to our full chat history
        st.session_state.messages.extend(filtered_messages)

async def main():
    st.markdown("<h1 style='text-align: right; direction: rtl;'>מדריך למשתמש טופס 101</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: right; direction: rtl;'>שאל כל שאלה בנוגע לטופס 101</p>", unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display ALL past messages in the UI
    for msg in st.session_state.messages:
        if isinstance(msg, (ModelRequest, ModelResponse)):
            for part in msg.parts:
                display_message_part(part)

    # Wait for user input
    user_input = st.chat_input("מה השאלה שלך?")

    if user_input:
        # 1) Add a new user message to the full conversation (for UI display)
        new_user_message = ModelRequest(parts=[UserPromptPart(content=user_input)])
        st.session_state.messages.append(new_user_message)

        # 2) Show that user message in the UI
        with st.chat_message("user"):
            st.markdown(f"<div dir='rtl'>{user_input}</div>", unsafe_allow_html=True)

        # 3) Build context with last 3 AI responses plus the new user message
        last_responses = get_last_ai_responses(st.session_state.messages)
        model_context = last_responses + [new_user_message]

        # 4) Stream the assistant's answer
        with st.chat_message("assistant"):
            await run_agent_with_streaming(user_input, model_context)

if __name__ == "__main__":
    asyncio.run(main())
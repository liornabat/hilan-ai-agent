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

# Get message history limit from environment variable, default to 2 if not set
MESSAGE_HISTORY_LIMIT = int(os.getenv("MESSAGE_HISTORY_LIMIT", 3))

# Initialize clients
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = Client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

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

def maintain_message_history(messages: list) -> list:
    """
    Return only the last N messages that are relevant for the model,
    EXCLUDING any 'tool' or 'tool-call' or 'tool-return' parts
    that can cause the 400 BadRequestError.
    """
    valid_messages = []
    tool_calls_pending = False

    for msg in messages:
        if not isinstance(msg, (ModelRequest, ModelResponse)):
            continue

        # Filter out parts with part_kind in {tool, tool-call, tool-return}
        # Also skip orphaned tool messages if no pending tool_calls.
        filtered_parts = []
        for part in msg.parts:
            # If there's a "tool_calls" property, it signals a tool usage
            if hasattr(part, 'tool_calls') and part.tool_calls:
                tool_calls_pending = True
            elif part.part_kind == 'tool' and not tool_calls_pending:
                # Skip orphaned tool messages
                continue
            elif part.part_kind == 'tool':
                # This completes the current tool call
                tool_calls_pending = False

            # Now exclude from the final LLM context any tool-like roles
            if part.part_kind not in {'tool', 'tool-call', 'tool-return'}:
                filtered_parts.append(part)

        # If no parts remain after filtering, skip the message entirely
        if not filtered_parts:
            continue

        # Create a new request/response object with filtered parts
        if isinstance(msg, ModelRequest):
            new_msg = ModelRequest(parts=filtered_parts)
        else:
            new_msg = ModelResponse(parts=filtered_parts)

        valid_messages.append(new_msg)

    # Keep only the last N valid messages
    if len(valid_messages) > MESSAGE_HISTORY_LIMIT:
        return valid_messages[-MESSAGE_HISTORY_LIMIT:]
    return valid_messages


def display_message_part(part):
    """
    Display a single part of a message in the Streamlit UI.
    We do NOT filter here, so the user sees the complete conversation,
    including any tool messages if they exist.
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
    elif part.part_kind in ('tool', 'tool-call', 'tool-return'):
        # Optional: Show them or skip them in the UI
        # If you want to see what the tool is doing, you can display them.
        # For now, let's skip showing these in the UI or just debug-log them.
        pass

async def run_agent_with_streaming(user_input: str, model_context: list):
    """
    Run the agent with streaming text support for RTL.
    'model_context' is the pruned (up to last N) set of messages
    that we actually send to the LLM as context.
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
            # Display partial text (assistant's streaming) in RTL
            message_placeholder.markdown(
                f"<div dir='rtl'>{partial_text}</div>",
                unsafe_allow_html=True
            )

        # Filter out user prompts from the new messages (we only keep assistant or system)
        filtered_messages = [
            msg for msg in result.new_messages()
            if not (
                hasattr(msg, 'parts') and
                any(part.part_kind == 'user-prompt' for part in msg.parts)
            )
        ]

        # Add the new assistant messages to the full chat history (no pruning for display)
        st.session_state.messages.extend(filtered_messages)

async def main():
    st.markdown("<h1 style='text-align: right; direction: rtl;'>מדריך למשתמש טופס 101</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: right; direction: rtl;'>שאל כל שאלה בנוגע לטופס 101</p>", unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []  # This stores the FULL conversation for display

    # 1) Display ALL messages in the UI (unfiltered)
    for msg in st.session_state.messages:
        if isinstance(msg, (ModelRequest, ModelResponse)):
            for part in msg.parts:
                display_message_part(part)

    # 2) Wait for user input
    user_input = st.chat_input("מה השאלה שלך?")

    if user_input:
        # Create new user message (store it unfiltered in session_state)
        new_message = ModelRequest(parts=[UserPromptPart(content=user_input)])
        st.session_state.messages.append(new_message)

        # Display the user's new message
        with st.chat_message("user"):
            st.markdown(f"<div dir='rtl'>{user_input}</div>", unsafe_allow_html=True)

        # 3) For the model context, we only want the last N messages
        #    *and* exclude any 'tool' parts.
        model_context = maintain_message_history(st.session_state.messages)

        # Display the assistant streaming response
        with st.chat_message("assistant"):
            await run_agent_with_streaming(user_input, model_context)

if __name__ == "__main__":
    asyncio.run(main())

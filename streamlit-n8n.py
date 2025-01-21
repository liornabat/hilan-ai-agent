from __future__ import annotations
from typing import Literal, TypedDict
import asyncio
import os
import streamlit as st
import aiohttp
from dotenv import load_dotenv

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

load_dotenv()

# Initialize clients
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = Client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# N8n webhook URL from environment variable
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL")


class N8nIntegration:
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        if not self.webhook_url:
            raise ValueError("N8N_WEBHOOK_URL environment variable not set")

    async def trigger_workflow(self, data: dict) -> dict:
        """
        Trigger n8n workflow asynchronously via webhook

        Args:
            data (dict): Data to send to n8n workflow

        Returns:
            dict: Response from n8n workflow
        """
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                        self.webhook_url,
                        json=data,
                        headers={'Content-Type': 'application/json'}
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        st.error(f"Error from n8n: {response.status}")
                        return None
            except Exception as e:
                st.error(f"Error triggering n8n workflow: {str(e)}")
                return None


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
    """Display a single part of a message in the Streamlit UI."""
    if part.part_kind == 'system-prompt':
        with st.chat_message("system"):
            st.markdown(f"**System**: {part.content}")
    elif part.part_kind == 'user-prompt':
        with st.chat_message("user"):
            st.markdown(part.content)
    elif part.part_kind == 'text':
        with st.chat_message("assistant"):
            st.markdown(part.content)

async def main():
    st.markdown("<h1 style='text-align: right; direction: rtl;'>מדריך למשתמש טופס 101</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: right; direction: rtl;'>שאל כל שאלה בנוגע לטופס 101</p>", unsafe_allow_html=True)

    # Initialize n8n integration
    n8n = N8nIntegration(N8N_WEBHOOK_URL)

    # Wait for user input
    user_input = st.chat_input("מה השאלה שלך?")

    if user_input:
        # Show user message in the UI
        with st.chat_message("user"):
            st.markdown(f"<div dir='rtl'>{user_input}</div>", unsafe_allow_html=True)

        # Stream the assistant's answer with n8n integration
        with st.chat_message("assistant"):
            await run_agent_with_streaming(user_input, n8n)


if __name__ == "__main__":
    asyncio.run(main())
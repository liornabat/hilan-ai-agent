import os
from dotenv import load_dotenv
import streamlit as st
import requests
import uuid
load_dotenv()
url = os.getenv("N8N_WEBHOOK_URL")
st.set_page_config(
    page_title="צ׳ט עם כל זכות",
)

# Add RTL support
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


def generate_session_id():
    return str(uuid.uuid4())


def send_message(session_id: str, user_input: str) -> str:
    """
    Send message to webhook and return the response
    """
    payload = {
        "sessionId": session_id,
        "chatInput": user_input
    }

    with st.spinner('מחכה לתשובה...'):
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()["output"]
        except Exception as e:
            st.error(f"Error sending message: {str(e)}")
            return "מצטערים, אירעה שגיאה בעת שליחת ההודעה. אנא נסה שוב."


def main():
    st.markdown("<h1 style='text-align: right; direction: rtl;'>צ׳ט עם כל זכות</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: right; direction: rtl;'>שאל כל שאלה בנוגע לאתר כל זכות </p>", unsafe_allow_html=True)

    # Initialize session state
    if "session_id" not in st.session_state:
        st.session_state.session_id = generate_session_id()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Add a reset button
    if st.button("התחל צ'אט חדש"):
        st.session_state.session_id = generate_session_id()
        st.session_state.messages = []
        st.rerun()

    # Display chat history
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        with st.chat_message(role):
            st.markdown(f"<div dir='rtl'>{content}</div>", unsafe_allow_html=True)

    # Get user input
    user_input = st.chat_input("מה השאלה שלך?")

    if user_input:
        # Display user message
        with st.chat_message("user"):
            st.markdown(f"<div dir='rtl'>{user_input}</div>", unsafe_allow_html=True)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Get and display assistant response with loading indicator
        assistant_response = send_message(st.session_state.session_id, user_input)
        with st.chat_message("assistant"):
            st.markdown(f"<div dir='rtl'>{assistant_response}</div>", unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})


if __name__ == "__main__":
    main()
# app.py
"""
Streamlit ChatGPT-like AI Chatbot Application (single-file)
- Uses st.chat_input() and st.chat_message() for a modern chat UI.
- Stores conversation in st.session_state["messages"] so history persists during the session.
- Uses OpenAI (from openai import OpenAI) if API key is provided; otherwise uses a simulated fallback assistant.
- Sidebar includes app title, description, model dropdown, API key input, and Clear Chat button.
- Run: streamlit run app.py
"""

import os
import time
import textwrap
import streamlit as st

# Required by your spec:
from openai import OpenAI  # Import present; we'll handle missing API key usage gracefully.

# --------------------------
# Helper functions
# --------------------------
def init_session():
    """Initialize session state storage for messages and settings."""
    if "messages" not in st.session_state:
        # messages: list of dicts {role: "user"|"assistant"|"system", "content": str}
        st.session_state["messages"] = []
    if "model_choice" not in st.session_state:
        st.session_state["model_choice"] = "General"
    if "api_key" not in st.session_state:
        st.session_state["api_key"] = os.getenv("OPENAI_API_KEY", "")


def append_message(role: str, content: str):
    """Append a message to the session message history."""
    st.session_state["messages"].append({"role": role, "content": content})


def clear_chat():
    """Clear the conversation history."""
    st.session_state["messages"] = []


def format_messages_for_openai():
    """Convert our internal messages to the OpenAI chat format."""
    formatted = []
    for m in st.session_state["messages"]:
        # Map our simple roles to the expected ones.
        role = m["role"]
        # role should be one of "user", "assistant", or "system"
        formatted.append({"role": role, "content": m["content"]})
    return formatted


def simulated_assistant_reply(user_message: str, model_choice: str) -> str:
    """
    Provide a deterministic local fallback reply when no OpenAI API key is available.
    This is intentionally simple but helpful: it detects a few common intents and
    returns a structured answer. It's not a real LLM, but it keeps the app functional
    and useful for prototyping or demos.
    """
    lower = user_message.strip().lower()

    # Basic intent heuristics
    if any(kw in lower for kw in ["explain", "what is", "define", "describe", "explanation"]):
        reply = (
            f"I don't have an external API key available, so here's a concise, local explanation of your request:\n\n"
            f"**Topic:** {user_message.strip()}\n\n"
            "**Short explanation:**\n"
            " - Start with a one-line summary.\n"
            " - Provide a simple example or analogy.\n"
            " - List 2–3 important facts or steps.\n\n"
            "If you'd like a more in-depth, GPT-powered explanation, add your OpenAI API key in the sidebar."
        )
        return reply

    if any(kw in lower for kw in ["how to", "steps to", "how do i", "how can i"]):
        reply = (
            f"You're asking for steps or instructions on: \"{user_message.strip()}\"\n\n"
            "Here's a practical step-by-step outline you can follow:\n"
            "1. Clarify the goal and inputs.\n"
            "2. Break the task into smaller sub-tasks.\n"
            "3. Implement each sub-task and test incrementally.\n"
            "4. Combine, test end-to-end, and iterate.\n\n"
            "Tip: If you want a runnable code sample or deeper explanation from a large model, provide an OpenAI API key in the sidebar."
        )
        return reply

    if any(kw in lower for kw in ["code", "implement", "script", "example", "function"]):
        # Attempt to provide a small illustrative snippet based on model_choice
        if "code" in lower or "python" in lower:
            snippet = textwrap.dedent(
                """
                # Example Python function - replace with specifics
                def greet(name: str) -> str:
                    \"\"\"Return a greeting for the given name.\"\"\"
                    return f"Hello, {name}! Welcome to your Streamlit chatbot demo."

                # Usage
                if __name__ == "__main__":
                    print(greet("Developer"))
                """
            )
            reply = (
                "I don't have an API key available, so here's a helpful local code example based on your request:\n\n"
                f"{snippet}\n\n"
                "If you want a tailored, longer code response from an LLM, add your OpenAI API key in the sidebar."
            )
            return reply
        else:
            reply = (
                "I see you're asking for a code example or implementation. Here's a general approach:\n\n"
                "1. Define the desired inputs and outputs.\n"
                "2. Write small functions for each step.\n"
                "3. Test each function with simple inputs.\n"
                "4. Combine the functions into a final script.\n\n"
                "Provide specifics (language, desired behavior) for something more concrete."
            )
            return reply

    # Default fallback: polite simulated answer with guidance
    reply = (
        "Simulated Assistant (no API key detected):\n\n"
        f"You asked: \"{user_message.strip()}\"\n\n"
        "Here's a helpful starting point:\n"
        " - I recommend clarifying what outcome you want (short answer, code sample, step-by-step guide, or explanation).\n"
        " - If you want quick facts, ask for them explicitly: e.g., \"Give me 3 key facts about X.\"\n\n"
        "When you're ready for more human-like, detailed responses, add an OpenAI API key in the sidebar."
    )
    return reply


def get_ai_response(api_key: str, model_choice: str) -> str:
    """
    Use OpenAI API if api_key is provided; otherwise return a simulated reply.
    This function expects that the user's last message is already appended to st.session_state["messages"].
    """
    # Ensure there is at least one user message
    if not st.session_state["messages"]:
        return "No messages in the conversation yet."

    # Find the last user message (walk from end)
    last_user_msg = None
    for msg in reversed(st.session_state["messages"]):
        if msg["role"] == "user":
            last_user_msg = msg["content"]
            break

    if last_user_msg is None:
        return "No recent user message found."

    # If an API key is provided, attempt to call the OpenAI API
    if api_key:
        try:
            # Initialize OpenAI client
            client = OpenAI(api_key=api_key)

            # Convert to the chat format expected by the OpenAI client
            messages = format_messages_for_openai()

            # Map our simple model choices to plausible model names (the exact model names may vary).
            # These are examples and can be adjusted by the user.
            model_map = {
                "General": "gpt-4o-mini",
                "Coding": "gpt-4o-mini-code",
                "Education": "gpt-4o-mini"
            }
            model_to_use = model_map.get(model_choice, "gpt-4o-mini")

            # Create the chat completion
            # NOTE: Different OpenAI clients / versions have slightly different method signatures.
            # We'll use the newer `client.chat.completions.create(...)` shape common in the OpenAI package.
            response = client.chat.completions.create(
                model=model_to_use,
                messages=messages,
                max_tokens=600,
            )

            # The structure returned may include choices with message content
            content = ""
            # Defensive parsing of response
            if getattr(response, "choices", None):
                # Try to collect assistant content
                parts = []
                for choice in response.choices:
                    # Some SDKs nest message -> content
                    msg = getattr(choice, "message", None) or choice.get("message", {})
                    part = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
                    if part:
                        parts.append(part)
                if parts:
                    content = "\n\n".join(parts)
            # fallback to text property
            if not content:
                # try other location
                content = getattr(response, "text", "") or str(response)

            if not content:
                content = "Received an empty response from OpenAI."

            return content

        except Exception as e:
            # API call failed: return a friendly message and a simulated fallback
            simulated = simulated_assistant_reply(last_user_msg, model_choice)
            return (
                f"⚠️ OpenAI API call failed: {getattr(e, 'message', str(e))}\n\n"
                "Falling back to the local simulated assistant:\n\n"
                f"{simulated}"
            )

    # No API key -> simulated assistant
    return simulated_assistant_reply(last_user_msg, model_choice)


# --------------------------
# Streamlit app UI
# --------------------------
def main():
    st.set_page_config(page_title="AI Chat Assistant", page_icon="🤖", layout="wide")
    init_session()

    # Sidebar
    with st.sidebar:
        st.title("AI Chat Assistant")
        st.markdown(
            "A lightweight ChatGPT-like Streamlit demo. "
            "You can optionally paste your OpenAI API key below to enable real LLM responses."
        )
        st.markdown("---")

        # Model dropdown (purely illustrative)
        model_choice = st.selectbox(
            "Model type",
            options=["General", "Coding", "Education"],
            index=["General", "Coding", "Education"].index(st.session_state.get("model_choice", "General")),
            key="sidebar_model_choice",
        )
        # Keep model_choice in session state for use elsewhere
        st.session_state["model_choice"] = model_choice

        st.markdown("**OpenAI API key**")
        api_key_input = st.text_input(
            "Paste your OpenAI API key here (optional)",
            value=st.session_state.get("api_key", ""),
            placeholder="sk-...",
            type="password",
            help="If provided, the app will attempt to use the OpenAI API. Otherwise a local simulated assistant will answer."
        )
        st.session_state["api_key"] = api_key_input.strip()

        st.markdown("---")
        if st.button("Clear Chat"):
            clear_chat()
            st.success("Chat history cleared.")

        st.markdown("---")
        st.caption("Built with Streamlit — demo app")

    # Main area: chat window
    st.header("AI Chat Assistant")
    st.write("Type a command, question, or description below. Conversation persists for the session.")

    # Display chat messages
    chat_container = st.container()
    with chat_container:
        # show all messages in order
        for msg in st.session_state["messages"]:
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(msg["content"])
            elif msg["role"] == "assistant":
                with st.chat_message("assistant"):
                    st.markdown(msg["content"])
            else:
                # system messages or others
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

    # Input area using st.chat_input
    user_input = st.chat_input("Type your message and press Enter...")
    if user_input:
        # Append user's message and display immediately
        append_message("user", user_input)

        # Show user's message right away in the UI
        with st.chat_message("user"):
            st.markdown(user_input)

        # Indicate assistant is thinking
        placeholder = st.empty()
        with placeholder.container():
            with st.chat_message("assistant"):
                st.markdown("...thinking...")

        # Get response (either via OpenAI API if key provided, or a local simulated reply)
        api_key = st.session_state.get("api_key", "").strip()
        model_choice = st.session_state.get("model_choice", "General")
        reply = get_ai_response(api_key=api_key, model_choice=model_choice)

        # Remove thinking placeholder and show actual reply
        placeholder.empty()
        append_message("assistant", reply)
        with st.chat_message("assistant"):
            st.markdown(reply)

    # Footer: small controls and export
    st.markdown("---")
    cols = st.columns([1, 1, 2])
    with cols[0]:
        if st.button("Download chat (txt)"):
            # Build a simple plain-text transcript
            transcript_lines = []
            for m in st.session_state["messages"]:
                role = m["role"].upper()
                transcript_lines.append(f"{role}: {m['content']}")
            transcript = "\n\n".join(transcript_lines)
            st.download_button("Click to download transcript", data=transcript, file_name="chat_transcript.txt", mime="text/plain")

    with cols[1]:
        if st.button("Copy last assistant message"):
            # Copy last assistant message to clipboard (browser-level copy not directly possible from server)
            last_assistant = ""
            for m in reversed(st.session_state["messages"]):
                if m["role"] == "assistant":
                    last_assistant = m["content"]
                    break
            if last_assistant:
                st.write("Here is the last assistant message — copy it below:")
                st.text_area("Last assistant message", value=last_assistant, height=200)
            else:
                st.info("No assistant message found to copy.")

    with cols[2]:
        st.caption("Tip: add your OpenAI API key in the sidebar to enable real GPT responses. This demo will still work without one.")

    # Keep the session_state updated and visible for debugging if developer flag enabled
    if st.checkbox("Show raw session state (debug)", value=False):
        st.write(st.session_state)


if __name__ == "__main__":
    main()

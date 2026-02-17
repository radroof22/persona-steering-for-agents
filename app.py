import streamlit as st

st.set_page_config(page_title="Overview", layout="wide")

st.title("Persona Steering Demo")

st.info(
    "This is a **technical demo** built on top of **Qwen 2.5-3B-Instruct**. "
    "It is not intended for production use. The goal is to demonstrate the power of "
    "**persona steering** — shaping a model's tone and personality by adding learned "
    "activation vectors to its hidden states at inference time, without any prompt "
    "engineering or finetuning.",
    icon="\u2139\ufe0f",
)

st.markdown(
    """
### Pages

- **Compare** — Type a prompt and see all personas respond side-by-side.
- **Chat** — Pick one persona and have a multi-turn conversation with it.

Use the sidebar to navigate.
"""
)

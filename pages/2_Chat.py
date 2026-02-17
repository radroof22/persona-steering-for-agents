import streamlit as st
from src.personas import PERSONAS
from src.activation_hooks import load_model, generate_steered
from src.persona_vectors import load_persona_vectors

st.set_page_config(page_title="Chat with Persona", layout="wide")


@st.cache_resource
def init_model():
    model, tokenizer, device = load_model("Qwen/Qwen2.5-3B-Instruct")
    return model, tokenizer, device


@st.cache_resource
def init_vectors():
    return load_persona_vectors("data/persona_vectors.pt")


# Load model and vectors
with st.spinner("Loading model and persona vectors..."):
    model, tokenizer, device = init_model()
    persona_vectors = init_vectors()

persona_lookup = {p["name"]: p for p in PERSONAS}
persona_names = [name for name in persona_vectors.keys() if name in persona_lookup]

# Sidebar controls
with st.sidebar:
    st.header("Persona Chat")

    selected = st.selectbox(
        "Persona",
        persona_names,
        format_func=lambda name: persona_lookup[name]["label"],
    )
    st.caption(persona_lookup[selected]["description"])

    st.markdown("---")
    alpha = st.slider("Steering strength (α)", min_value=0.0, max_value=15.0, value=4.0, step=0.5)
    layer_start = st.slider("Start layer", min_value=0, max_value=35, value=18)
    layer_end = st.slider("End layer", min_value=0, max_value=35, value=24)
    layers = list(range(layer_start, layer_end + 1))
    max_tokens = st.slider("Max tokens", min_value=50, max_value=300, value=150, step=25)

    st.markdown("---")
    if st.button("Clear chat"):
        st.session_state.chat_messages = []
        st.rerun()

# Reset chat when persona changes
if "chat_persona" not in st.session_state:
    st.session_state.chat_persona = selected
    st.session_state.chat_messages = []

if st.session_state.chat_persona != selected:
    st.session_state.chat_persona = selected
    st.session_state.chat_messages = []

st.title(f"Chat — {persona_lookup[selected]['label']}")

# Display existing messages
for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if user_input := st.chat_input("Type your message..."):
    # Add user message
    st.session_state.chat_messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Build conversation prompt
    prompt_parts = []
    for msg in st.session_state.chat_messages:
        if msg["role"] == "user":
            prompt_parts.append(f"Customer: {msg['content']}")
        else:
            prompt_parts.append(f"Agent: {msg['content']}")
    prompt_parts.append("Agent:")
    prompt = "\n".join(prompt_parts)

    # Generate steered response
    with st.chat_message("assistant"):
        with st.spinner("Generating..."):
            response = generate_steered(
                model, tokenizer, prompt, persona_vectors[selected],
                alpha=alpha, layers=layers,
                max_new_tokens=max_tokens, device=device,
            )
        st.markdown(response)

    st.session_state.chat_messages.append({"role": "assistant", "content": response})

import streamlit as st
from src.personas import PERSONAS
from src.activation_hooks import load_model, generate_all_personas
from src.persona_vectors import load_persona_vectors

st.set_page_config(page_title="Compare Personas", layout="wide")


@st.cache_resource
def init_model():
    model, tokenizer, device = load_model("Qwen/Qwen2.5-3B-Instruct")
    return model, tokenizer, device


@st.cache_resource
def init_vectors():
    return load_persona_vectors("data/persona_vectors.pt")


st.title("Compare Personas")
st.markdown(
    "Type a customer service prompt and see how the **same model** responds differently "
    "when steered with different persona vectors."
)

# Sidebar controls
with st.sidebar:
    st.header("Steering Controls")
    alpha = st.slider("Steering strength (α)", min_value=0.0, max_value=15.0, value=4.0, step=0.5)
    layer_start = st.slider("Start layer", min_value=0, max_value=35, value=18)
    layer_end = st.slider("End layer", min_value=0, max_value=35, value=24)
    layers = list(range(layer_start, layer_end + 1))
    max_tokens = st.slider("Max tokens", min_value=50, max_value=300, value=150, step=25)

    st.markdown("---")
    st.markdown(f"**Layers:** {layers}")
    st.markdown(f"**α:** {alpha}")

# Load model and vectors
with st.spinner("Loading model and persona vectors..."):
    model, tokenizer, device = init_model()
    persona_vectors = init_vectors()

st.success(f"Model loaded on `{device}` | {len(persona_vectors)} persona vectors ready")

# Prompt input
prompt = st.text_area(
    "Customer service prompt",
    value="Customer: My order hasn't arrived and it's been two weeks. I'm really frustrated.\nAgent:",
    height=100,
)

persona_lookup = {p["name"]: p for p in PERSONAS}

if st.button("Generate", type="primary"):
    with st.spinner("Generating baseline + all 9 personas in one batch..."):
        results = generate_all_personas(
            model, tokenizer, prompt, persona_vectors,
            alpha=alpha, layers=layers,
            max_new_tokens=max_tokens, device=device,
        )

    # Baseline
    st.subheader("Baseline (no steering)")
    st.markdown(f"> {results['baseline']}")
    st.divider()

    # Steered responses in a grid
    persona_names = [name for name in persona_vectors.keys() if name in persona_lookup]
    cols_per_row = 3
    rows = [persona_names[i:i + cols_per_row] for i in range(0, len(persona_names), cols_per_row)]

    for row in rows:
        cols = st.columns(len(row))
        for col, persona_name in zip(cols, row):
            info = persona_lookup[persona_name]
            with col:
                st.subheader(info["label"])
                st.caption(info["description"])
                st.markdown(results.get(persona_name, "*(no output)*"))

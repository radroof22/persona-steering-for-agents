# Sentosa — Persona Steering Demo

A technical demo that shapes a language model's tone and personality by adding learned activation vectors to its hidden states at inference time — no prompt engineering or fine-tuning required.

Built on top of **Qwen 2.5-3B-Instruct**.

## How It Works

1. **Generate contrastive data** — For each persona, generate paired responses: one with a "trait" system prompt and one with an "anti-trait" prompt using DeepInfra's API.
2. **Compute steering vectors** — Extract hidden-state activations for each pair, compute per-layer difference vectors (`mean(trait) - mean(anti-trait)`), and normalize.
3. **Steer at inference** — Register PyTorch forward hooks that add the persona vector to the model's hidden states during generation: `hidden[:, -1, :] += alpha * vector`.

## Personas

| Persona | Style |
|---------|-------|
| Urgent & Direct | Cuts to the chase, leads with the fix |
| Warm & Patient | Step-by-step, welcoming, never rushed |
| Polished & Premium | White-glove, proactive, VIP treatment |
| Technical & Precise | Exact steps, version numbers, specs |
| Empathetic & Compassionate | Acknowledges feelings before solving |
| Friendly & Conversational | Personable, uses humor, builds rapport |
| Value-Focused | Surfaces deals and savings proactively |
| Gentle & Simple | Short sentences, simple language, kind |
| Cautious & Thorough | Covers caveats, edge cases, fine print |

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Create a `.env` file:

```
DEEPINFRA_API_KEY=<your-key>
```

## Usage

### Streamlit App

```bash
streamlit run app.py
```

Two pages are available:

- **Compare** — Type a prompt and see all 9 personas respond side-by-side.
- **Chat** — Pick a persona and have a multi-turn conversation.

### Notebooks

- `notebooks/01_persona_data_generation.ipynb` — Generate contrastive training pairs via DeepInfra.
- `notebooks/02_persona_steering.ipynb` — Compute steering vectors and test them interactively.

## Project Structure

```
app.py                  # Streamlit entry point
pages/
  1_Compare.py          # Side-by-side persona comparison
  2_Chat.py             # Multi-turn chat with one persona
src/
  activation_hooks.py   # PyTorch hooks for steering during generation
  api_client.py         # DeepInfra API client for data generation
  persona_vectors.py    # Vector computation and persistence
  personas.py           # Persona definitions and evaluation prompts
data/
  persona_responses.json  # Pre-generated contrastive pairs
  persona_vectors.pt      # Pre-computed steering vectors
notebooks/
  01_persona_data_generation.ipynb
  02_persona_steering.ipynb
```

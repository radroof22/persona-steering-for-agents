import json
import torch
from pathlib import Path

from src.activation_hooks import extract_activations, _get_layers


def compute_persona_vectors(model, tokenizer, device: str, responses_path: str = "data/persona_responses.json") -> dict[str, dict[int, torch.Tensor]]:
    """Compute persona vectors from saved contrastive response pairs.

    For each persona and each layer:
        v_l = mean(trait_activations) - mean(anti_trait_activations)

    Returns dict mapping persona_name -> {layer_idx: vector}.
    """
    with open(responses_path) as f:
        responses = json.load(f)

    persona_vectors = {}
    num_layers = len(_get_layers(model))

    for persona_name, prompts in responses.items():
        print(f"Computing vectors for: {persona_name}")
        trait_acts = {i: [] for i in range(num_layers)}
        anti_acts = {i: [] for i in range(num_layers)}

        skipped = 0
        for prompt_key, data in prompts.items():
            # Extract activations for trait response
            t_acts = extract_activations(model, tokenizer, data["trait_response"], device)
            a_acts = extract_activations(model, tokenizer, data["anti_response"], device)

            # Skip pairs where either response is empty
            if t_acts is None or a_acts is None:
                skipped += 1
                continue

            for layer_idx, act in t_acts.items():
                trait_acts[layer_idx].append(act)
            for layer_idx, act in a_acts.items():
                anti_acts[layer_idx].append(act)

        # Compute mean difference per layer
        vectors = {}
        for layer_idx in range(num_layers):
            trait_mean = torch.stack(trait_acts[layer_idx]).mean(dim=0)
            anti_mean = torch.stack(anti_acts[layer_idx]).mean(dim=0)
            vec = trait_mean - anti_mean
            # Normalize
            vec = vec / (vec.norm() + 1e-8)
            vectors[layer_idx] = vec

        persona_vectors[persona_name] = vectors
        print(f"  -> Done ({len(prompts)} pairs, {skipped} skipped, {num_layers} layers)")

    return persona_vectors


def save_persona_vectors(persona_vectors: dict[str, dict[int, torch.Tensor]], path: str = "data/persona_vectors.pt"):
    """Save persona vectors to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # Convert to serializable format
    serializable = {}
    for name, layer_vecs in persona_vectors.items():
        serializable[name] = {layer_idx: vec for layer_idx, vec in layer_vecs.items()}
    torch.save(serializable, path)
    print(f"Saved persona vectors to {path}")


def load_persona_vectors(path: str = "data/persona_vectors.pt") -> dict[str, dict[int, torch.Tensor]]:
    """Load persona vectors from disk."""
    return torch.load(path, weights_only=True)

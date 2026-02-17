import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _get_layers(model):
    """Get the list of transformer layers regardless of architecture."""
    # Qwen2, Llama, Mistral, Gemma, Phi — all use model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    # GPT-2 uses model.transformer.h
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError(f"Unknown model architecture: {type(model).__name__}")


def _get_hidden_states(output):
    """Extract hidden states from layer output, handling both tuple and tensor formats."""
    if isinstance(output, tuple):
        return output[0]
    return output


# Stop generation when the model tries to continue the conversation as the customer
_TURN_STOPS = ["Customer:", "customer:", "Supplier:", "Agent:", "\nUser:", "\nHuman:"]


def _trim_to_single_turn(text: str) -> str:
    """Truncate at the first sign of a new conversation turn."""
    earliest = len(text)
    for stop in _TURN_STOPS:
        idx = text.find(stop)
        if idx != -1 and idx < earliest:
            earliest = idx
    return text[:earliest].rstrip()


def load_model(model_name: str = "Qwen/Qwen2.5-3B-Instruct", device: str = None):
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer, device


def extract_activations(model, tokenizer, text: str, device: str) -> dict[int, torch.Tensor] | None:
    """Extract residual stream activations at the last token for each layer.

    Returns dict mapping layer_index -> activation tensor of shape (hidden_dim,),
    or None if the text is empty / produces no tokens.
    """
    if not text or not text.strip():
        return None

    activations = {}
    layers = _get_layers(model)

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            hidden = _get_hidden_states(output)
            activations[layer_idx] = hidden[0, -1, :].detach().float().cpu()
        return hook_fn

    hooks = []
    for i, block in enumerate(layers):
        hooks.append(block.register_forward_hook(make_hook(i)))

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    if inputs["input_ids"].shape[1] == 0:
        for h in hooks:
            h.remove()
        return None

    with torch.no_grad():
        model(**inputs)

    for h in hooks:
        h.remove()

    return activations


class SteeringHook:
    """Context manager that adds a persona vector to activations during generation."""

    def __init__(self, model, persona_vectors: dict[int, torch.Tensor], alpha: float = 1.0, layers: list[int] = None):
        self.model = model
        self.persona_vectors = persona_vectors
        self.alpha = alpha
        self.layers = layers or list(persona_vectors.keys())
        self._hooks = []
        self._model_layers = _get_layers(model)

    def _make_hook(self, layer_idx):
        vec = self.persona_vectors[layer_idx].to(dtype=torch.float16, device=self.model.device)

        def hook_fn(module, input, output):
            hidden = _get_hidden_states(output)
            hidden[:, -1, :] += self.alpha * vec
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden

        return hook_fn

    def __enter__(self):
        for layer_idx in self.layers:
            if layer_idx in self.persona_vectors:
                hook = self._model_layers[layer_idx].register_forward_hook(
                    self._make_hook(layer_idx)
                )
                self._hooks.append(hook)
        return self

    def __exit__(self, *args):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


class BatchSteeringHook:
    """Context manager that steers each batch item with a different persona vector.

    steering_matrix: dict[int, Tensor] mapping layer_idx -> (batch_size, hidden_dim)
        Row 0 can be zeros for baseline, rows 1..N are persona vectors.
    """

    def __init__(self, model, steering_matrix: dict[int, torch.Tensor], alpha: float = 1.0, layers: list[int] = None):
        self.model = model
        self.steering_matrix = steering_matrix
        self.alpha = alpha
        self.layers = layers or list(steering_matrix.keys())
        self._hooks = []
        self._model_layers = _get_layers(model)

    def _make_hook(self, layer_idx):
        mat = self.steering_matrix[layer_idx].to(dtype=torch.float16, device=self.model.device)

        def hook_fn(module, input, output):
            hidden = _get_hidden_states(output)
            # mat is (batch_size, hidden_dim), add to last token of each batch item
            hidden[:, -1, :] += self.alpha * mat
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden

        return hook_fn

    def __enter__(self):
        for layer_idx in self.layers:
            if layer_idx in self.steering_matrix:
                hook = self._model_layers[layer_idx].register_forward_hook(
                    self._make_hook(layer_idx)
                )
                self._hooks.append(hook)
        return self

    def __exit__(self, *args):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def generate_steered(model, tokenizer, prompt: str, persona_vectors: dict[int, torch.Tensor],
                     alpha: float = 1.0, layers: list[int] = None, max_new_tokens: int = 200,
                     device: str = "cpu") -> str:
    """Generate text with persona steering applied (single persona)."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with SteeringHook(model, persona_vectors, alpha=alpha, layers=layers):
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            stop_strings=_TURN_STOPS,
            tokenizer=tokenizer,
        )

    generated = output[0][inputs["input_ids"].shape[1]:]
    return _trim_to_single_turn(tokenizer.decode(generated, skip_special_tokens=True))


def generate_baseline(model, tokenizer, prompt: str, max_new_tokens: int = 200, device: str = "cpu") -> str:
    """Generate text without any steering."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
        stop_strings=_TURN_STOPS,
        tokenizer=tokenizer,
    )

    generated = output[0][inputs["input_ids"].shape[1]:]
    return _trim_to_single_turn(tokenizer.decode(generated, skip_special_tokens=True))


def generate_all_personas(model, tokenizer, prompt: str,
                          persona_vectors_dict: dict[str, dict[int, torch.Tensor]],
                          alpha: float = 1.0, layers: list[int] = None,
                          max_new_tokens: int = 200, device: str = "cpu") -> dict[str, str]:
    """Generate baseline + all persona-steered responses in a single batched pass.

    Returns dict mapping "baseline" and each persona name to its generated text.
    """
    persona_names = list(persona_vectors_dict.keys())
    batch_size = 1 + len(persona_names)  # baseline + personas
    layers = layers or list(next(iter(persona_vectors_dict.values())).keys())

    # Tokenize and repeat prompt for the full batch
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"].repeat(batch_size, 1)
    attention_mask = inputs["attention_mask"].repeat(batch_size, 1)
    prompt_len = inputs["input_ids"].shape[1]

    # Build steering matrix: row 0 = zeros (baseline), rows 1..N = persona vectors
    hidden_dim = next(iter(next(iter(persona_vectors_dict.values())).values())).shape[0]
    steering_matrix = {}
    for layer_idx in layers:
        mat = torch.zeros(batch_size, hidden_dim)
        for i, name in enumerate(persona_names):
            if layer_idx in persona_vectors_dict[name]:
                mat[i + 1] = persona_vectors_dict[name][layer_idx]
        steering_matrix[layer_idx] = mat

    # Generate all at once
    with BatchSteeringHook(model, steering_matrix, alpha=alpha, layers=layers):
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode each batch item and trim to single agent turn
    results = {}
    for i, output in enumerate(outputs):
        generated = output[prompt_len:]
        text = _trim_to_single_turn(tokenizer.decode(generated, skip_special_tokens=True))
        if i == 0:
            results["baseline"] = text
        else:
            results[persona_names[i - 1]] = text

    return results

import os
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def get_client():
    return OpenAI(
        api_key=os.environ["DEEPINFRA_API_KEY"],
        base_url="https://api.deepinfra.com/v1/openai",
    )


def generate_response(client, system_prompt: str, user_prompt: str, model: str = "openai/gpt-oss-120b-Turbo") -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=512,
        temperature=0.7,
    )
    return response.choices[0].message.content


def generate_contrastive_pairs(client, personas: list, prompts: list, model: str = "openai/gpt-oss-120b-Turbo", save_path: str = "data/persona_responses.json"):
    """Generate contrastive response pairs for all personas and prompts.

    Saves incrementally so progress isn't lost if interrupted.
    """
    save_file = Path(save_path)
    save_file.parent.mkdir(parents=True, exist_ok=True)

    # Load existing progress
    if save_file.exists():
        with open(save_file) as f:
            results = json.load(f)
    else:
        results = {}

    total = len(personas) * len(prompts)
    done = 0

    for persona in personas:
        name = persona["name"]
        if name not in results:
            results[name] = {}

        for i, prompt in enumerate(prompts):
            prompt_key = f"prompt_{i}"

            if prompt_key in results[name]:
                done += 1
                continue

            trait_response = generate_response(client, persona["system_prompt"], prompt, model)
            anti_response = generate_response(client, persona["anti_system_prompt"], prompt, model)

            results[name][prompt_key] = {
                "prompt": prompt,
                "trait_response": trait_response,
                "anti_response": anti_response,
            }

            done += 1

            # Save after each pair
            with open(save_file, "w") as f:
                json.dump(results, f, indent=2)

            if done % 10 == 0:
                print(f"Progress: {done}/{total}")

    print(f"Done! {done}/{total} pairs saved to {save_path}")
    return results

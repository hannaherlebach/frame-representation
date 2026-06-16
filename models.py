"""Model clients for the MMLU evaluation pipeline."""

import os
import time

import anthropic

MODEL_CONFIGS = {
    "claude-sonnet-4-5": {
        "provider": "anthropic",
        "model_id": "claude-sonnet-4-5",
        # Pricing in USD per million tokens (https://www.anthropic.com/pricing)
        # Thinking tokens are billed at the output rate.
        "cost_per_million": {"input": 3.00, "output": 15.00},
    },
    # ── Open-weights models (cluster) ─────────────────────────────────────────
    # Served via vLLM on FLAIR. Set VLLM_BASE_URL=http://localhost:8000/v1
    # after port-forwarding from the cluster (see Dockerfile.vllm + serve_llama.sh).
    "llama-3.1-8b": {
        "provider": "vllm",
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "cost_per_million": {"input": 0.0, "output": 0.0},  # gated — needs HF approval
    },
    "qwen2.5-7b": {
        "provider": "vllm",
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "cost_per_million": {"input": 0.0, "output": 0.0},  # open — no approval needed
    },
    # "qwen3-32b": {
    #     "provider": "vllm",
    #     "model_id": "Qwen/Qwen3-32B",
    #     "cost_per_million": {"input": 0.0, "output": 0.0},
    # },
}

_clients: dict = {}


def _get_client(provider: str):
    if provider not in _clients:
        if provider == "anthropic":
            _clients[provider] = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

        elif provider == "vllm":
            from openai import OpenAI
            _clients[provider] = OpenAI(
                api_key="EMPTY",  # vLLM doesn't require a real key
                base_url=os.environ["VLLM_BASE_URL"],
            )

    return _clients[provider]


def query_model(
    model_name: str,
    user_prompt: str,
    system_prompt: str | None = None,
    thinking_budget: int | None = None,
    cot: bool = False,
    max_retries: int = 3,
) -> tuple[str, str | None, int, int]:
    """Query the model and return (response_text, thinking_text, input_tokens, output_tokens).

    Prefill (default): constrained to a single letter via assistant turn prefill.
    CoT: model reasons visibly in output, must end with "Answer: X" (2048 output tokens).
    Thinking: extended thinking block + short text answer.
    """
    config = MODEL_CONFIGS[model_name]
    provider = config["provider"]
    model_id = config["model_id"]
    client = _get_client(provider)

    for attempt in range(max_retries):
        try:
            if provider == "anthropic":
                if thinking_budget is not None:
                    # Extended thinking: model reasons internally, then outputs "Answer: X".
                    # Prefilling is incompatible with thinking, so we rely on the prompt.
                    kwargs: dict = {
                        "model": model_id,
                        "max_tokens": thinking_budget + 1024,
                        "thinking": {"type": "enabled", "budget_tokens": thinking_budget},
                        "messages": [{"role": "user", "content": user_prompt}],
                    }
                    if system_prompt:
                        kwargs["system"] = system_prompt
                    response = client.messages.create(**kwargs)
                    thinking = next((b.thinking for b in response.content if b.type == "thinking"), None)
                    text = next(b.text for b in response.content if b.type == "text")
                elif cot:
                    # CoT: model reasons in output text, no prefill, 2048 output tokens.
                    kwargs = {
                        "model": model_id,
                        "max_tokens": 2048,
                        "messages": [{"role": "user", "content": user_prompt}],
                    }
                    if system_prompt:
                        kwargs["system"] = system_prompt
                    response = client.messages.create(**kwargs)
                    thinking = None
                    text = response.content[0].text.strip()
                else:
                    # Prefill forces the model to complete from "Answer:" rather
                    # than starting a chain-of-thought.
                    kwargs = {
                        "model": model_id,
                        "max_tokens": 4,
                        "messages": [
                            {"role": "user", "content": user_prompt},
                            {"role": "assistant", "content": "Answer:"},
                        ],
                    }
                    if system_prompt:
                        kwargs["system"] = system_prompt
                    response = client.messages.create(**kwargs)
                    thinking = None
                    text = response.content[0].text.strip()

                return (text, thinking, response.usage.input_tokens, response.usage.output_tokens)

            elif provider == "vllm":
                # vLLM serves an OpenAI-compatible chat API. No prefill or
                # extended thinking — the prompt template handles answer format.
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": user_prompt})
                response = client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    max_tokens=2048 if cot else 16,
                )
                return (
                    response.choices[0].message.content.strip(),
                    None,  # no thinking traces for open-weights models
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                )

        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            print(f"  [retry {attempt + 1}/{max_retries}] {e} — waiting {wait}s")
            time.sleep(wait)

    raise RuntimeError("Unreachable")

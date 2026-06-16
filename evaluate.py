"""MMLU abstract algebra data loading and model evaluation."""

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from datasets import concatenate_datasets, load_dataset
from tqdm import tqdm

from models import MODEL_CONFIGS, query_model

CHOICE_LABELS = ["A", "B", "C", "D"]

# Used when prefilling "Answer:" — model just completes the letter.
QUESTION_TEMPLATE = """\
{question}

A) {A}
B) {B}
C) {C}
D) {D}

Answer with only the letter (A, B, C, or D). You MUST NOT include any other text or reasoning; you must answer DIRECTLY with ONLY your answer letter."""

# Used with extended thinking — model reasons in the thinking block, text output is just the answer.
QUESTION_TEMPLATE_THINKING = """\
{question}

A) {A}
B) {B}
C) {C}
D) {D}

Reply only with your answer in the format:
Answer: X
where X is one of A, B, C, or D."""

# Used for chain-of-thought mode — model reasons visibly in its output, then commits to "Answer: X".
QUESTION_TEMPLATE_COT = """\
{question}

A) {A}
B) {B}
C) {C}
D) {D}

Think through this step by step. You must end your response with your answer on its own line, \
exactly in the format:
Answer: X
where X is one of A, B, C, or D."""


@dataclass
class EvalResult:
    accuracy: float
    stderr: float       # standard error of the mean (std / sqrt(n))
    input_tokens: int
    output_tokens: int
    cost_usd: float
    completions: list[dict]

    def __str__(self) -> str:
        return (
            f"accuracy={self.accuracy:.1%} ±{self.stderr:.1%}  "
            f"tokens={self.input_tokens:,}in/{self.output_tokens:,}out  "
            f"cost=${self.cost_usd:.4f}"
        )


def load_subjects(subjects: list[str], split: str = "test"):
    """Load and concatenate one or more MMLU subjects."""
    splits = [load_dataset("cais/mmlu", s, split=split) for s in subjects]
    return concatenate_datasets(splits) if len(splits) > 1 else splits[0]


def format_question(sample: dict, thinking: bool = False, cot: bool = False) -> str:
    choices = dict(zip(CHOICE_LABELS, sample["choices"]))
    if thinking:
        template = QUESTION_TEMPLATE_THINKING
    elif cot:
        template = QUESTION_TEMPLATE_COT
    else:
        template = QUESTION_TEMPLATE
    return template.format(question=sample["question"], **choices)


def parse_answer(response: str) -> str | None:
    """Extract the answer letter from the model response.

    Handles two formats:
    - Prefill responses: response is just the completion after "Answer:", e.g. " B"
    - Thinking responses: response contains "Answer: B" somewhere in the text
    """
    upper = response.strip().upper()
    if upper in CHOICE_LABELS:
        return upper
    match = re.search(r'ANSWER:\s*([A-D])', upper)
    return match.group(1) if match else None


def run_evaluation(
    model_name: str,
    dataset,
    system_prompt: str | None = None,
    thinking_budget: int | None = None,
    cot: bool = False,
    limit: int | None = None,
    log_path: Path | None = None,
) -> EvalResult:
    """Evaluate a model on the dataset; returns an EvalResult with accuracy and cost."""
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    correct = 0
    total_input_tokens = 0
    total_output_tokens = 0
    completions = []

    # vLLM can't prefill, so use the "Answer: X" template for its standard mode
    provider = MODEL_CONFIGS[model_name]["provider"]
    use_answer_format = provider == "vllm" and not cot and thinking_budget is None

    for sample in tqdm(dataset, desc=model_name, leave=False):
        prompt = format_question(
            sample,
            thinking=thinking_budget is not None or use_answer_format,
            cot=cot,
        )
        text, thinking, in_tok, out_tok = query_model(
            model_name, prompt, system_prompt, thinking_budget=thinking_budget, cot=cot
        )
        total_input_tokens += in_tok
        total_output_tokens += out_tok
        predicted = parse_answer(text)
        correct_label = CHOICE_LABELS[sample["answer"]]
        is_correct = predicted == correct_label
        if is_correct:
            correct += 1
        entry = {
            "system_prompt": system_prompt,
            "prompt": prompt,
            "thinking": thinking,
            "response": text,
            "predicted": predicted,
            "correct_label": correct_label,
            "correct": is_correct,
        }
        completions.append(entry)
        if log_path is not None:
            with open(log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")

    scores = [1 if c["correct"] else 0 for c in completions]
    n = len(scores)
    p = correct / n
    std = math.sqrt(sum((s - p) ** 2 for s in scores) / (n - 1)) if n > 1 else 0.0
    stderr = std / math.sqrt(n)

    pricing = MODEL_CONFIGS[model_name]["cost_per_million"]
    cost = (
        total_input_tokens / 1e6 * pricing["input"]
        + total_output_tokens / 1e6 * pricing["output"]
    )

    return EvalResult(
        accuracy=p,
        stderr=stderr,
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        cost_usd=cost,
        completions=completions,
    )

"""
Persona system prompt templates for MMLU experiments.
Modify these strings to change how each persona is framed.
Add or remove entries to expand the set of personas.
"""

PROMPTS: dict[str, str] = {
    "low_pressure": (
        "You are a {persona}. Answer )"
    ),

    "medium_pressure": (
        "You are a {persona}.)"
    ),

    "high_pressure": (
        "You are a {persona}.)"
    ),
}

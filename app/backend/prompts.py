from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .config import MAX_CONTEXT_CHARS
from .vectorstores import Chunk


def build_context(chunks: Iterable[Chunk]) -> str:
    context_parts: list[str] = []
    total = 0
    for chunk in chunks:
        header = f"[Source: {Path(chunk.source).name}, page {chunk.page}]\n"
        block = header + chunk.text.strip() + "\n"
        if total + len(block) > MAX_CONTEXT_CHARS:
            break
        context_parts.append(block)
        total += len(block)
    return "\n".join(context_parts)


def build_chat_prompt(
    query: str,
    chunks: Iterable[Chunk],
    language: str = "en",
    show_steps: bool = False,
    mode: str = "answer",
    user_solution: str = "",
) -> tuple[str, str]:
    context = build_context(chunks)
    system = (
        "You are an expert Electrical & Electronic Engineering tutor. "
        "Always cite sources using the [Source: file, page] format. "
        "Be precise with equations, units, and steps. "
        "If the user asks to check a solution, identify errors and provide corrections."
    )
    instructions = [
        f"Language: {'Bengali' if language == 'bn' else 'English'}.",
        "Use the provided context for factual claims.",
        "If something is not in context, say it is not found in the sources.",
        f"Provide step-by-step reasoning: {'yes' if show_steps else 'no'}.",
    ]
    if mode == "check":
        instructions.append("The user provided a solution. Evaluate and correct it.")

    user_prompt = query
    if user_solution:
        user_prompt += f"\n\nUser solution:\n{user_solution}"

    prompt = (
        "CONTEXT:\n"
        f"{context}\n\n"
        "INSTRUCTIONS:\n"
        + "\n".join(instructions)
        + "\n\nUSER QUESTION:\n"
        + user_prompt
    )
    return system, prompt

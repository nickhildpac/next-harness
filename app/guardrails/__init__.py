"""Two-wall guardrail engine for the agent loop.

Input validation catches prompt injection and PII before the LLM sees the goal.
Output validation catches unsafe content, leaked secrets, and PII before the user
sees the result. Every task run passes through both walls; nothing passes unchecked.

Findings carry an ``action``:

- ``"blocked"`` — hard stop. Prompt injection blocks a run before it reaches the
  model; unsafe content or leaked secrets replace the delivered result. These are
  categories that must never pass a wall.
- ``"redacted"`` — the offending span (PII) is replaced with a placeholder and the
  sanitized text flows on, so a stray email or phone number doesn't fail the task.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# --- Detection rules ---------------------------------------------------------
# Each rule is (label, compiled_pattern). PII rules also carry a replacement token.
# Patterns bound their gap matches (e.g. ``[^.\n]{0,40}``) to avoid runaway backtracking.

_INJECTION_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "instruction override",
        re.compile(
            r"\b(ignore|disregard|forget)\b[^.\n]{0,40}"
            r"\b(previous|prior|above|earlier|all)\b[^.\n]{0,20}"
            r"\b(instructions?|prompts?|rules?|context|messages?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "system prompt exfiltration",
        re.compile(
            r"\b(reveal|show|print|repeat|expose|leak|display)\b[^.\n]{0,40}"
            r"\b(system\s+prompt|initial\s+instructions?|your\s+instructions?|your\s+prompt)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "safety override",
        re.compile(
            r"\b(ignore|bypass|override|disable|turn\s+off)\b[^.\n]{0,30}"
            r"\b(safety|guardrails?|filters?|restrictions?|policy|policies|rules?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "jailbreak persona",
        re.compile(
            r"\b(do\s+anything\s+now|developer\s+mode|jailbreak|dan\s+mode|"
            r"unfiltered\s+mode|no\s+restrictions)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "injected role marker",
        re.compile(
            r"<\|?\s*(im_start|im_end|system|assistant)\s*\|?>|^\s*###\s*system\b",
            re.IGNORECASE | re.MULTILINE,
        ),
    ),
)

_PII_RULES: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "email address",
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        "[REDACTED_EMAIL]",
    ),
    (
        "social security number",
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "[REDACTED_SSN]",
    ),
    (
        "credit card number",
        re.compile(r"\b(?:\d[ -]?){15}\d\b"),
        "[REDACTED_CC]",
    ),
    (
        "phone number",
        re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"),
        "[REDACTED_PHONE]",
    ),
)

_SECRET_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("openai-style api key", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    ("aws access key id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("private key block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    (
        "credential assignment",
        re.compile(
            r"(?i)\b(api[_-]?key|secret|token|passwd|password)\b\s*[:=]\s*\S{6,}",
        ),
    ),
)

_UNSAFE_OUTPUT_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "weapon or explosive instructions",
        re.compile(
            r"(?i)\b(how\s+to|instructions?|steps?|guide|recipe)\b[^.\n]{0,40}"
            r"\b(bomb|explosive|nerve\s+agent|bioweapon|chemical\s+weapon|"
            r"meth(?:amphetamine)?)\b",
        ),
    ),
    (
        "malware development",
        re.compile(
            r"(?i)\b(build|create|write|develop|deploy|code)\b[^.\n]{0,25}"
            r"\b(ransomware|keylogger|rootkit|botnet|spyware)\b",
        ),
    ),
    (
        "self-harm instructions",
        re.compile(
            r"(?i)\b(how\s+to|best\s+way\s+to|methods?\s+to|ways?\s+to)\b[^.\n]{0,25}"
            r"\b(kill\s+yourself|commit\s+suicide|end\s+your\s+life|self[- ]harm)\b",
        ),
    ),
)


SAFE_OUTPUT_REPLACEMENT = (
    "This response was withheld: it triggered an output guardrail for unsafe or sensitive content."
)


@dataclass(frozen=True)
class GuardrailFinding:
    category: str  # "prompt_injection" | "pii" | "unsafe_content" | "secret"
    detail: str  # human-readable label, e.g. "email address"
    action: str  # "blocked" | "redacted"


@dataclass(frozen=True)
class GuardrailResult:
    allowed: bool
    text: str
    findings: tuple[GuardrailFinding, ...] = field(default_factory=tuple)

    @property
    def redacted(self) -> bool:
        return any(f.action == "redacted" for f in self.findings)

    @property
    def blocked_reason(self) -> str | None:
        blocks = [f for f in self.findings if f.action == "blocked"]
        if not blocks:
            return None
        return "; ".join(f"{f.category}: {f.detail}" for f in blocks)


def _redact(text: str) -> tuple[str, list[GuardrailFinding]]:
    findings: list[GuardrailFinding] = []
    for label, pattern, replacement in _PII_RULES:
        if pattern.search(text):
            findings.append(GuardrailFinding("pii", label, "redacted"))
            text = pattern.sub(replacement, text)
    return text, findings


def _scan_blocking(
    text: str, category: str, rules: tuple[tuple[str, re.Pattern[str]], ...]
) -> list[GuardrailFinding]:
    return [
        GuardrailFinding(category, label, "blocked")
        for label, pattern in rules
        if pattern.search(text)
    ]


class Guardrails:
    """The two walls. ``check_input`` guards the goal; ``check_output`` guards the result."""

    def __init__(self, *, enabled: bool = True):
        self.enabled = enabled

    def check_input(self, text: str) -> GuardrailResult:
        """Wall 1: prompt injection blocks the run; PII is redacted before the LLM sees it."""
        if not self.enabled or not text:
            return GuardrailResult(allowed=True, text=text)
        findings = _scan_blocking(text, "prompt_injection", _INJECTION_RULES)
        sanitized, pii = _redact(text)
        findings.extend(pii)
        allowed = not any(f.action == "blocked" for f in findings)
        return GuardrailResult(allowed=allowed, text=sanitized, findings=tuple(findings))

    def check_output(self, text: str) -> GuardrailResult:
        """Wall 2: unsafe content or leaked secrets block delivery; PII is redacted."""
        if not self.enabled or not text:
            return GuardrailResult(allowed=True, text=text)
        findings = _scan_blocking(text, "unsafe_content", _UNSAFE_OUTPUT_RULES)
        findings.extend(_scan_blocking(text, "secret", _SECRET_RULES))
        sanitized, pii = _redact(text)
        findings.extend(pii)
        allowed = not any(f.action == "blocked" for f in findings)
        return GuardrailResult(allowed=allowed, text=sanitized, findings=tuple(findings))


__all__ = [
    "Guardrails",
    "GuardrailResult",
    "GuardrailFinding",
    "SAFE_OUTPUT_REPLACEMENT",
]

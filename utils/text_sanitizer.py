"""
Basic text sanitization utilities for incident descriptions.

Goals:
- Remove dangerous control characters.
- Strip simple HTML tags.
- Normalize whitespace and markdown noise.
- Softly neutralize obvious prompt-injection markers.
- Optionally clamp very long inputs.
"""

import re
from typing import Optional


class TextSanitizer:
    """Sanitizes free-text reports before AI processing."""

    def __init__(self, max_length: int = 4000):
        """
        Args:
            max_length: Maximum number of characters to keep after sanitization.
        """
        self.max_length = max_length
        # Simple regexes compiled once
        self._html_tag_re = re.compile(r"<[^>]+>")
        self._control_re = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
        self._markdown_re = re.compile(r"[*_`#]+")
        self._url_re = re.compile(r"https?://\S+")

        # Very simple prompt-injection style phrases to soften
        self._pi_markers = [
            "ignore previous instructions",
            "forget previous instructions",
            "you are chatgpt",
            "you are an ai model",
            "follow these instructions",
            "system prompt:",
        ]

    def _neutralize_prompt_markers(self, text: str) -> str:
        """
        Replace obvious instruction phrases with a neutralized form
        while keeping the surrounding narrative readable.
        """
        lowered = text.lower()
        for marker in self._pi_markers:
            if marker in lowered:
                # Replace case-insensitively by tagging as narrative text
                pattern = re.compile(re.escape(marker), re.IGNORECASE)
                text = pattern.sub(f"[{marker} (reported speech)]", text)
        return text

    def sanitize(self, text: Optional[str]) -> str:
        """Return a cleaned version of the input text."""
        if not text:
            return ""

        cleaned = str(text)

        # Remove control characters (except newline and tab)
        cleaned = self._control_re.sub("", cleaned)

        # Strip simple HTML tags like <script>, <b>, etc.
        cleaned = self._html_tag_re.sub(" ", cleaned)

        # Drop bare URLs (keep as generic token)
        cleaned = self._url_re.sub(" [URL] ", cleaned)

        # Remove basic markdown formatting characters
        cleaned = self._markdown_re.sub(" ", cleaned)

        # Neutralize obvious prompt-injection markers
        cleaned = self._neutralize_prompt_markers(cleaned)

        # Normalize whitespace (including multiple newlines)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        # Clamp to max_length to avoid pathological inputs
        if len(cleaned) > self.max_length:
            cleaned = cleaned[: self.max_length]

        return cleaned



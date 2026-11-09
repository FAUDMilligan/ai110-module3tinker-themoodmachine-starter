# mood_analyzer.py
"""
Rule based mood analyzer for short text snippets.

This class starts with very simple logic:
  - Preprocess the text
  - Look for positive and negative words
  - Compute a numeric score
  - Convert that score into a mood label
"""

import re
from typing import List, Dict, Tuple, Optional

from dataset import POSITIVE_WORDS, NEGATIVE_WORDS

# Matches one token: a text emoticon, a single emoji, or a word.
# Text is lowercased before matching, so ":D" arrives as ":d".
TOKEN_PATTERN = re.compile(
    r"[:;=]-?[()\[\]dp/\\|]"                       # emoticons like :) :-( ;p :/
    r"|[\U0001F300-\U0001FAFF☀-➿]"        # emojis like 😂 💀 🥲 ☹
    r"|[a-z0-9]+(?:'[a-z]+)*"                      # words, keeping can't / i'm
)

# Words that flip the meaning of the sentiment word right after them.
NEGATORS = {"not", "never", "no", "isn't", "wasn't", "don't", "didn't", "ain't"}

# Emojis, emoticons, and slang that carry a strong mood on their own.
# 💀 and 😂 are scored positive because they usually mean "dying of laughter".
# 🥲 is left out on purpose: it can mean happy, sad, or both.
SIGNAL_WEIGHTS = {
    ":)": 2, ":-)": 2, ":d": 2, "😂": 2, "💀": 1, "lol": 1,
    ":(": -2, ":-(": -2, "☹": -2, "😭": -1,
}

# Situations people almost never mean it when they say they love them.
# A positive word next to one of these is treated as sarcasm.
ANNOYING_SITUATIONS = [
    "stuck in traffic",
    "traffic",
    "group project",
    "mondays",
    "waiting in line",
    "on hold",
    "flight delayed",
    "delayed flight",
    "my alarm",
]


class MoodAnalyzer:
    """
    A very simple, rule based mood classifier.
    """

    def __init__(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
    ) -> None:
        # Use the default lists from dataset.py if none are provided.
        positive_words = positive_words if positive_words is not None else POSITIVE_WORDS
        negative_words = negative_words if negative_words is not None else NEGATIVE_WORDS

        # Store as sets for faster lookup.
        self.positive_words = set(w.lower() for w in positive_words)
        self.negative_words = set(w.lower() for w in negative_words)

    # ---------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------

    def preprocess(self, text: str) -> List[str]:
        """
        Convert raw text into a list of tokens the model can work with.

        Steps:
          - Strip whitespace and lowercase everything
          - Normalize curly apostrophes so "can’t" matches "can't"
          - Shorten letters repeated 3+ times ("soooo" -> "soo")
          - Keep text emoticons (":)", ":-(", ":D") as single tokens
          - Split emojis ("🥲", "😂", "💀") into their own tokens,
            even when they are stuck to a word or to each other
          - Drop all other punctuation, keeping apostrophes inside words
        """
        cleaned = text.strip().lower().replace("’", "'")
        cleaned = re.sub(r"([a-z])\1{2,}", r"\1\1", cleaned)
        return TOKEN_PATTERN.findall(cleaned)

    # ---------------------------------------------------------------------
    # Scoring logic
    # ---------------------------------------------------------------------

    def score_text(self, text: str) -> int:
        """
        Compute a numeric "mood score" for the given text.

        Positive words increase the score.
        Negative words decrease the score.

        Modeling improvements:
          - Negation: a negator right before a sentiment word flips it,
            so "not happy" counts as negative and "not bad" as positive.
          - Strong signals: emojis, emoticons, and slang in SIGNAL_WEIGHTS
            add their own weight (for example ":)" is +2, ":(" is -2).
          - Counting: every occurrence counts, so "💀💀" counts twice.
        """
        score, _, _ = self._analyze(text)
        return score

    def _analyze(self, text: str) -> Tuple[int, List[str], List[str]]:
        """
        Shared scoring loop used by score_text, predict_label, and explain.

        Returns (score, positive_hits, negative_hits). A negated word is
        recorded with its negator, for example "not happy" in negative_hits.
        """
        tokens = self.preprocess(text)
        score = 0
        positive_hits: List[str] = []
        negative_hits: List[str] = []

        for i, token in enumerate(tokens):
            if token in SIGNAL_WEIGHTS:
                value = SIGNAL_WEIGHTS[token]
                label = token
            elif token in self.positive_words or token in self.negative_words:
                value = 1 if token in self.positive_words else -1
                label = token
                if i > 0 and tokens[i - 1] in NEGATORS:
                    value = -value
                    label = f"{tokens[i - 1]} {token}"
            else:
                continue

            score += value
            if value > 0:
                positive_hits.append(label)
            else:
                negative_hits.append(label)

        return score, positive_hits, negative_hits

    # ---------------------------------------------------------------------
    # Label prediction
    # ---------------------------------------------------------------------

    def predict_label(self, text: str) -> str:
        """
        Turn the numeric score for a piece of text into a mood label.

        Mapping:
          - positive word + annoying situation         -> "sarcasm"
          - both positive and negative signals present -> "mixed"
          - score > 0  -> "positive"
          - score < 0  -> "negative"
          - score == 0 -> "neutral"
        """
        score, positive_hits, negative_hits = self._analyze(text)

        if positive_hits and self._mentions_annoying_situation(text):
            return "sarcasm"
        if positive_hits and negative_hits:
            return "mixed"
        if score > 0:
            return "positive"
        if score < 0:
            return "negative"
        return "neutral"

    def _mentions_annoying_situation(self, text: str) -> bool:
        """Return True if the text contains a phrase from ANNOYING_SITUATIONS."""
        joined = f" {' '.join(self.preprocess(text))} "
        return any(f" {phrase} " in joined for phrase in ANNOYING_SITUATIONS)

    # ---------------------------------------------------------------------
    # Explanations (optional but recommended)
    # ---------------------------------------------------------------------

    def explain(self, text: str) -> str:
        """
        Return a short string explaining WHY the model chose its label.

        Example:
          'Label = mixed, score = 1 (positive: ['fun', '💀'], negative: ['bad'])'
        """
        score, positive_hits, negative_hits = self._analyze(text)
        label = self.predict_label(text)

        return (
            f"Label = {label}, score = {score} "
            f"(positive: {positive_hits}, negative: {negative_hits})"
        )

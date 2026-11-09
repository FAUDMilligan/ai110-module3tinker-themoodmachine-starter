"""
Shared data for the Mood Machine lab.

This file defines:
  - POSITIVE_WORDS: starter list of positive words
  - NEGATIVE_WORDS: starter list of negative words
  - SAMPLE_POSTS: short example posts for evaluation and training
  - TRUE_LABELS: human labels for each post in SAMPLE_POSTS
"""

# ---------------------------------------------------------------------
# Starter word lists
# ---------------------------------------------------------------------

POSITIVE_WORDS = [
    "happy",
    "great",
    "good",
    "love",
    "excited",
    "awesome",
    "fun",
    "chill",
    "relaxed",
    "amazing",
]

NEGATIVE_WORDS = [
    "sad",
    "bad",
    "terrible",
    "awful",
    "angry",
    "upset",
    "tired",
    "stressed",
    "hate",
    "boring",
]

# ---------------------------------------------------------------------
# Starter labeled dataset
# ---------------------------------------------------------------------

# Short example posts written as if they were social media updates or messages.
SAMPLE_POSTS = [
    "I love this class so much",
    "Today was a terrible day",
    "Feeling tired but kind of hopeful",
    "This is fine",
    "So excited for the weekend",
    "I am not happy about this",
    "Lowkey stressed but kind of proud of myself",
    "I absolutely love getting stuck in traffic",
    "no cap this pizza is amazing 😂",
    "Got a B on the exam 🥲",
    "highkey hate mondays :(",
    "Just finished my homework",
    "That movie was so bad it was actually fun 💀",
    "Great, another group project. Can't wait.",
    "chillin with friends :)",
    "I'm dead 💀💀 this is the funniest thing ever",
    "ugh my phone died again :(",
    "not bad for a monday",
    "I'm so tired of this lol",
    "lowkey love this weather",
    "Can't believe I passed!! 😭",
]

# Human labels for each post above.
# Allowed labels in the starter:
#   - "positive"
#   - "negative"
#   - "neutral"
#   - "mixed"
#   - "sarcasm" (positive words used to mean something negative)
TRUE_LABELS = [
    "positive",  # "I love this class so much"
    "negative",  # "Today was a terrible day"
    "mixed",     # "Feeling tired but kind of hopeful"
    "neutral",   # "This is fine"
    "positive",  # "So excited for the weekend"
    "negative",  # "I am not happy about this"
    "mixed",     # "Lowkey stressed but kind of proud of myself"
    "sarcasm",   # "I absolutely love getting stuck in traffic"
    "positive",  # "no cap this pizza is amazing 😂"
    "mixed",     # "Got a B on the exam 🥲" (edge case: could be read as positive)
    "negative",  # "highkey hate mondays :("
    "neutral",   # "Just finished my homework" (edge case: could be relief/positive)
    "positive",  # "That movie was so bad it was actually fun 💀" (edge case: "bad" + "fun")
    "sarcasm",   # "Great, another group project. Can't wait."
    "positive",  # "chillin with friends :)"
    "positive",  # "I'm dead 💀💀 this is the funniest thing ever" (slang: "dead" = laughing)
    "negative",  # "ugh my phone died again :("
    "positive",  # "not bad for a monday" (negation: "not bad")
    "negative",  # "I'm so tired of this lol" (edge case: "lol" here is exasperated, not happy)
    "positive",  # "lowkey love this weather"
    "positive",  # "Can't believe I passed!! 😭" (edge case: 😭 as happy tears)
]

# Edge cases worth inspecting (people might disagree on these):
#   - "Got a B on the exam 🥲": 🥲 can mean relief, disappointment, or both.
#   - "Just finished my homework": neutral fact, or quietly positive relief?
#   - "That movie was so bad it was actually fun 💀": mixed or positive?
#   - Sarcastic posts ("I absolutely love...", "Great, ... Can't wait.") use
#     positive words to say something negative, which should trip up the
#     rule-based model.
#   - "I'm dead 💀💀": negative-sounding slang that actually means laughing.

assert len(SAMPLE_POSTS) == len(TRUE_LABELS)

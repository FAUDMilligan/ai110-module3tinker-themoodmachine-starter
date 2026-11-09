# Model Card: Mood Machine

This model card covers **both** versions of the Mood Machine mood classifier:

1. A **rule based model** in `mood_analyzer.py`
2. A **machine learning model** in `ml_experiments.py` (scikit-learn)

## 1. Model Overview

**Model type:**
I compared both models on the same labeled posts.

**Intended purpose:**
Classify short, informal social-media-style posts into one of five moods: `positive`, `negative`, `neutral`, `mixed`, or `sarcasm`. It is a classroom exercise for seeing how and why text classifiers fail, not a tool for real decisions about people.

**How it works (brief):**
- **Rule based:** clean the text into tokens, add up points for positive words, negative words, emojis and slang, flip a word's points if "not"/"never"/etc. comes right before it, then turn the score into a label (details in Section 3).
- **ML:** turn each post into word counts (bag of words), then train a logistic regression classifier on the labeled posts so it learns which words go with which label.

## 2. Data

**Dataset description:**
`SAMPLE_POSTS` has **21 posts**. The starter had 6. I added 10 that cover slang ("lowkey", "highkey", "no cap"), emojis and emoticons (😂 💀 🥲 `:)` `:(`), sarcasm, and mixed feelings. During evaluation I added 5 more (phone dying, "not bad for a monday", "tired of this lol", "lowkey love this weather", "Can't believe I passed!! 😭") to see which model was more sensitive to new data.

**Labeling process:**
Each post got one label for its overall mood. I added a `sarcasm` label partway through, because labeling "I absolutely love getting stuck in traffic" as `negative` hid the reason it is negative. Posts that could reasonably get a different label:

- "Got a B on the exam 🥲" → `mixed`. 🥲 can mean relief, disappointment, or both. Someone else could say `positive`.
- "Just finished my homework" → `neutral`. Could be read as relief (`positive`).
- "That movie was so bad it was actually fun 💀" → `positive`. Could easily be `mixed`.
- "I'm so tired of this lol" → `negative`. "lol" here is exasperated, not happy.
- "Can't believe I passed!! 😭" → `positive`. 😭 is used as happy crying here.

**Important characteristics of the dataset:**
- Contains slang, emojis, and text emoticons
- Includes 2 sarcastic posts
- 3 posts express mixed feelings
- Posts are all one sentence and very short

**Possible issues with the dataset:**
- **Tiny:** 21 posts is far too few to train or fairly test anything.
- **Imbalanced:** 9 positive, 5 negative, 3 mixed, 2 neutral, 2 sarcasm.
- **One voice:** every post and label reflects one writer's style and reading. The slang is mostly young, US, internet English. There is no AAVE, regional slang, other languages, or code-switching.
- **No test set:** both models are evaluated on the same posts they were built or trained on.

## 3. How the Rule Based Model Works

**Preprocessing (`preprocess`):**
- Lowercase, and convert curly apostrophes (’) to straight ones.
- Shorten letters repeated 3+ times ("soooo" → "soo").
- Keep text emoticons (`:)`, `:-(`, `:D`) as single tokens, and split every emoji into its own token, even when attached to a word ("amazing😂").
- Drop other punctuation but keep apostrophes inside words ("can't", "i'm").

**Scoring rules (`score_text`):**
- Each word in `POSITIVE_WORDS` adds +1, and each word in `NEGATIVE_WORDS` adds −1. Repeats count every time.
- **Negation (main enhancement):** if the token right before a sentiment word is a negator (`not`, `never`, `no`, `isn't`, `don't`, …), the word's sign flips. "not happy" = −1, "not bad" = +1.
- **Strong signals:** a `SIGNAL_WEIGHTS` table scores emojis, emoticons, and slang directly: `:)` `:D` 😂 = +2, 💀 `lol` = +1, `:(` ☹ = −2, 😭 = −1. 🥲 is left unscored on purpose because it is ambiguous.

**Label thresholds (`predict_label`), checked in this order:**
1. A positive signal **and** a phrase from `ANNOYING_SITUATIONS` ("stuck in traffic", "group project", "mondays", …) → `sarcasm`
2. Both positive and negative signals present → `mixed`
3. Score > 0 → `positive`; score < 0 → `negative`; score = 0 → `neutral`

`explain()` shows the label, the score, and exactly which tokens counted, using the same scoring code as `predict_label`.

**Strengths:**
- Predictable and explainable: every decision traces back to specific tokens.
- Handles simple negation ("I am not happy about this", "not bad for a monday").
- Understands emojis and emoticons, which the ML model ignores completely (see Section 4).
- Works the same way on sentences it has never seen.

**Weaknesses:**
- Only knows the words on its lists. "hopeful" and "proud" aren't on the positive list, so two `mixed` posts look purely `negative`.
- Each emoji has one fixed meaning. 😭 is always sad and 💀 is always laughing, but real use depends on context.
- The sarcasm rule is a keyword hack. "I love mondays" is called sarcasm even if someone means it.
- "Any positive + any negative = mixed" is too blunt. "so bad it was actually fun" is really positive.

## 4. How the ML Model Works

**Features used:**
Bag of words using scikit-learn's default `CountVectorizer`. Its default tokenizer keeps only words of 2+ letters or digits, so **all emojis and emoticons are thrown away**, along with "I" and "B". The learned vocabulary has no 😂, 💀, 🥲, `:)`, or `:(` in it.

**Training data:**
Trained on `SAMPLE_POSTS` and `TRUE_LABELS` from `dataset.py`, using `LogisticRegression(max_iter=1000)`.

**Training behavior:**
- **Before** adding the 5 new posts, I trained on the original 16 and asked it to predict the 5 new ones. It got **1 of 5** right and predicted `positive` for almost everything, including "ugh my phone died again :(" (the `:(` is invisible to it) and "I'm so tired of this lol".
- **After** adding those 5 posts to the training data, it got **all 21** right, including all 5 new ones. The model didn't get smarter: it memorized the new examples.

**Strengths:**
- Learns from labels automatically, with no hand-written rules. It picked up `mixed` for "tired but kind of hopeful" and "stressed but … proud" without anyone adding "hopeful" or "proud" to a list.
- Will copy any labeling convention, including `sarcasm`.

**Weaknesses:**
- Overfits a dataset this small: 100% on training posts, near-guessing on new ones.
- Blind to emojis and emoticons, which carry most of the mood in posts like "Got a B on the exam 🥲".
- Can pick up spurious cues. For example "traffic", "group", "project", and "mondays" become `sarcasm`/`negative` words only because of one or two posts.

## 5. Evaluation

**How I evaluated:**
Both models were scored on the labeled posts in `dataset.py` with `python main.py` and `python ml_experiments.py`. Both evaluations reuse posts the models were built or trained on, so these numbers are optimistic.

| Dataset | Rule based | ML (training accuracy) |
|---|---|---|
| Original 16 posts | 12/16 = **0.75** | 16/16 = **1.00** |
| 5 new posts, before ML retrains | 3/5 | 1/5 |
| All 21 posts | 15/21 = **0.71** | 21/21 = **1.00** |

**Examples of correct predictions (rule based):**
- "I am not happy about this" → `negative`. Negation flipped "happy" from +1 to −1.
- "highkey hate mondays :(" → `negative`. "hate" (−1) + `:(` (−2) = −3.
- "I'm dead 💀💀 this is the funniest thing ever" → `positive`. Each 💀 is +1. "dead" isn't on the negative list, so slang doesn't trip it.

**Rule-based mismatches (6 of 21), grouped by pattern:**

| Pattern | Post | True | Predicted | Why |
|---|---|---|---|---|
| Missing vocabulary | Feeling tired but kind of hopeful | mixed | negative | "hopeful" isn't a positive word, so only "tired" (−1) counts |
| Missing vocabulary | Lowkey stressed but kind of proud of myself | mixed | negative | "proud" isn't a positive word |
| Emoji meaning depends on context | Got a B on the exam 🥲 | mixed | neutral | 🥲 is deliberately unscored, so the score is 0 |
| Emoji meaning depends on context | Can't believe I passed!! 😭 | positive | negative | 😭 is scored −1, but here it means happy tears |
| Mixed rule too blunt | That movie was so bad it was actually fun 💀 | positive | mixed | "bad" + "fun" triggers `mixed` |
| Mixed rule too blunt / slang tone | I'm so tired of this lol | negative | mixed | "lol" is scored as happy, but here it is exasperated |

Every miss involves either **words the lists don't cover** or **emoji/slang whose meaning flips with context**. Plain, literal sentences are all correct.

**Reasoning walkthrough of one wrong prediction: "Can't believe I passed!! 😭"**
1. `preprocess` → `["can't", "believe", "i", "passed", "😭"]`
2. "can't", "believe", "i", "passed" aren't in any word list or `SIGNAL_WEIGHTS`, so they add 0. "passed" carries the actual good news, but the model has no idea.
3. "😭" is in `SIGNAL_WEIGHTS` as −1, so the score is −1 with negative hits `['😭']`.
4. There are no positive hits, so `sarcasm` and `mixed` are skipped. Score < 0 → `negative`.

The logic did exactly what it was written to do. The failure is a modeling assumption, "😭 always means sad", which this sentence breaks.
**Decision: documented as a limitation, not fixed.** Changing 😭 to positive would just break the posts where it really means sad, and "passed" is only positive in some contexts.

**Do the two models fail in the same places?**
No. On the training posts the ML model has no failures, because it memorized them. On the 5 unseen posts, the models failed on different things:
- **The rule model** handled `:(` and "not bad" correctly but misread 😭 and "lol".
- **The ML model** missed "ugh my phone died again :(", because it can't see `:(` and "phone"/"died" were new words. It also called "tired of this lol" `positive`.

Neither handled "tired of this lol" correctly.

**Which model changed more when I added data?** The **ML model**. On those 5 posts it went from 1/5 to 5/5 once they were in its training data. The rule model's behavior didn't change at all, because its rules don't depend on the data. Only its accuracy number moved, since there were new posts to score. The ML model is far more sensitive to the data and labels.

## 6. Limitations

- **Sarcasm is only caught by keyword.** "I love getting stuck in traffic" is labeled `sarcasm` only because "stuck in traffic" is on a hard-coded list. "I love when my code breaks right before the deadline" would be labeled `positive`, because "love" scores +1 and that situation isn't on the list. In the other direction, someone who sincerely says "I love mondays" is wrongly labeled `sarcasm`.
- **Unknown words are invisible.** "Feeling tired but kind of hopeful" was labeled `negative` because "tired" scored −1 and "hopeful" isn't in `POSITIVE_WORDS`.
- **Each emoji has one fixed meaning.** "Can't believe I passed!! 😭" was labeled `negative` because 😭 is always −1.
- **Negation only looks one word back.** "not very happy" is still counted as positive, because "very" sits between "not" and "happy".
- **Tiny, single-author dataset with no held-out test set.** The 0.71 and 1.00 accuracies say little about real-world performance. The ML model's 1.00 is pure memorization.
- **The ML model ignores emojis entirely**, because of `CountVectorizer`'s default tokenizer.

## 7. Ethical Considerations

**Overt bias note:**
Errors cluster around **informal, emoji-heavy, internet-slang writing**. All 6 rule-based misses involve emojis, slang tone ("lol"), or casual phrasing. Plain, standard-English sentences were all labeled correctly. The model **underserves people whose emotional language is expressed through emoji and slang**, which skews younger, and anyone whose dialect isn't in the word lists:
- The word lists are short, standard-English words.
- Emoji meanings were fixed by one person's reading. For example, "💀 = laughing" is generational and would be wrong for many older users.
- There is no AAVE, regional slang, or non-English text, so those writers would mostly get `neutral` or wrong labels.

**Other risks:**
- **Missing real distress:** "I'm fine lol 💀" would be scored `positive`. A real mood-monitoring system built like this could miss someone who is struggling but jokes about it.
- **Privacy:** analyzing personal messages for mood without consent is a privacy problem, even when the model is accurate.

**Covert compliance note (did my AI assistant push back?):**
I used an AI coding assistant (Claude) to write the new posts, labels, preprocessing, scoring rules, and thresholds. What it did and didn't flag:
- **Flagged (coverage gaps):** "hopeful" and "proud" are missing from `POSITIVE_WORDS`. The two sarcastic posts would fail a word-counting model. The keyword sarcasm rule would mislabel sincere "I love mondays". 🥲 is too ambiguous to score. `predict_label` was still a stub when my commit message said it was done.
- **Not flagged (bias):** it never raised that the word lists, slang, and emoji meanings reflect one narrow (young, US, internet) English style. It also never pointed out that it was choosing the labels and emoji meanings itself, so its own reading became the dataset's "ground truth".
- **Just agreed:** when I asked it to make "I love getting stuck in traffic" return `sarcasm`, it implemented a keyword rule right away. It noted the false-positive risk but didn't push back on whether a keyword list is a real sarcasm detector.

**It should have pushed back on those two points before writing the data.** Its coverage-gap warnings were useful, but it treated bias as a word-list completeness problem, not a question of whose language the model is built around.

## 8. Ideas for Improvement

- Add many more labeled posts from different writers, and have two people label each post to measure disagreement.
- Hold out a real test set instead of reporting training accuracy.
- **Rule based:** add "hopeful", "proud", "passed", etc. Use a 2–3 word negation window ("not very happy"). Only call a post `mixed` when the score is near 0. Let emoji weights depend on nearby words.
- **ML:** use a tokenizer that keeps emojis and emoticons (pass `preprocess` as the `CountVectorizer` tokenizer), try TF-IDF, and add word pairs (`ngram_range=(1, 2)`) so "not happy" is learned as one unit.
- Try a small pretrained transformer for sarcasm and context-dependent emoji.

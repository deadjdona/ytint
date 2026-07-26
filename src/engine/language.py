"""Corpus-level language detection and multilingual stopword assembly.

Fixes a pipeline bug where s02_topics.py hardcoded English stopwords
(`CountVectorizer(stop_words="english")`) regardless of actual comment
language, while s01_enrich.py runs a Russian-only sentiment model
(`blanchefort/rubert-base-cased-sentiment`). On non-English corpora this
leaves zero stopword filtering in the c-TF-IDF topic labels, polluting
every cluster name with function words (и, на, что, ...).

No language is assumed. Dominant languages are detected from a corpus
sample and their stopword sets merged, so this works unmodified whether
the channel is Russian, Ukrainian, English, or code-switched.
"""

from __future__ import annotations

import numpy as np
from langdetect import DetectorFactory, LangDetectException, detect_langs
from stopwordsiso import langs as supported_langs
from stopwordsiso import stopwords as iso_stopwords

DetectorFactory.seed = 42

_PLATFORM_NOISE = frozenset({
    "http", "https", "www", "com", "youtube", "youtu", "amp", "gt", "lt",
    "quot", "nbsp",
})

_SUPPORTED = supported_langs()
_MIN_CHARS = 3


def detect_corpus_languages(
    docs: list[str],
    sample_size: int = 3000,
    min_share: float = 0.05,
    random_state: int = 42,
) -> set[str]:
    """Detect dominant language(s) present in a text corpus."""
    rng = np.random.default_rng(random_state)
    if len(docs) > sample_size:
        idx = rng.choice(len(docs), size=sample_size, replace=False)
        sample = [docs[i] for i in idx]
    else:
        sample = docs

    counts: dict[str, int] = {}
    n_detected = 0
    for text in sample:
        text = (text or "").strip()
        if len(text) < _MIN_CHARS:
            continue
        try:
            top = detect_langs(text)[0]
        except LangDetectException:
            continue
        if top.prob < 0.5:
            continue
        counts[top.lang] = counts.get(top.lang, 0) + 1
        n_detected += 1

    if n_detected == 0:
        return {"en"}

    return {lang for lang, c in counts.items() if c / n_detected >= min_share}


def build_stopword_set(languages: set[str]) -> frozenset[str]:
    """Merge stopword sets for all detected languages plus platform noise."""
    usable = languages & _SUPPORTED
    words = set(_PLATFORM_NOISE)
    if usable:
        words |= iso_stopwords(list(usable))
    return frozenset(words)


def corpus_stopwords(docs: list[str], **detect_kwargs) -> tuple[frozenset[str], set[str]]:
    """Detect languages then build the merged stopword set."""
    languages = detect_corpus_languages(docs, **detect_kwargs)
    return build_stopword_set(languages), languages

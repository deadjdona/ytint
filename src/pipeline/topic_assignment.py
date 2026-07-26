"""Helpers for preserving the canonical comments layer during topic modeling."""

from collections.abc import Sequence

import pandas as pd


def attach_topics(
    comments: pd.DataFrame,
    modeled_comments: pd.DataFrame,
    topics: Sequence[int],
) -> pd.DataFrame:
    """Return all comments with topics assigned only to rows used for modeling.

    Reply rows are intentionally retained and receive the BERTopic outlier value
    ``-1`` because they were excluded from semantic clustering.
    """
    if len(modeled_comments) != len(topics):
        raise ValueError(
            "The number of topic assignments must match the modeled comments."
        )

    result = comments.copy()
    result["topic"] = -1
    result.loc[modeled_comments.index, "topic"] = list(topics)
    return result

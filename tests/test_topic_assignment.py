import pandas as pd

from pipeline.topic_assignment import attach_topics


def test_attach_topics_preserves_reply_rows_and_assigns_only_modeled_comments():
    comments = pd.DataFrame(
        {
            "comment_id": ["top-1", "reply-1", "top-2"],
            "parent_id": [None, "top-1", None],
            "text": ["first", "reply", "second"],
        }
    )
    modeled_comments = comments.iloc[[0, 2]]

    result = attach_topics(comments, modeled_comments, [7, 9])

    assert result["comment_id"].tolist() == ["top-1", "reply-1", "top-2"]
    assert result.loc[1, "parent_id"] == "top-1"
    assert result.loc[[0, 2], "parent_id"].isna().all()
    assert result["topic"].tolist() == [7, -1, 9]


def test_attach_topics_rejects_mismatched_assignment_count():
    comments = pd.DataFrame({"comment_id": ["top-1"], "parent_id": [None]})

    try:
        attach_topics(comments, comments, [])
    except ValueError as error:
        assert "topic assignments" in str(error)
    else:
        raise AssertionError("Expected mismatched topic assignments to raise ValueError")

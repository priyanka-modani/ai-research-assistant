from research_assistant.retrieval.fusion import reciprocal_rank_fusion


def test_rrf_rewards_items_present_in_both_rankings():
    result = reciprocal_rank_fusion([["a", "b", "c"], ["b", "d", "a"]])
    ids = [item_id for item_id, _ in result]
    assert ids[0] in {"a", "b"}
    assert ids.index("a") < ids.index("c")
    assert ids.index("b") < ids.index("d")

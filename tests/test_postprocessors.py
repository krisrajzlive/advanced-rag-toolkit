from llama_index.core.schema import NodeWithScore, TextNode

from src.retrieval.postprocessors import (
    apply_chain,
    long_context_reorder,
    out_of_scope_rate,
    similarity_cutoff,
)


def _nodes(*scores):
    return [NodeWithScore(node=TextNode(text=f"n{i}"), score=s) for i, s in enumerate(scores)]


def test_similarity_cutoff_drops_weak_matches():
    kept = apply_chain([similarity_cutoff(0.3)], _nodes(0.9, 0.31, 0.29, 0.1), "q")
    assert [n.score for n in kept] == [0.9, 0.31]


def test_apply_chain_does_not_mutate_input_nodes():
    nodes = _nodes(0.9, 0.5)

    class Rescore:  # stands in for a reranker that overwrites scores in place
        def postprocess_nodes(self, nodes, query_bundle):
            for n in nodes:
                n.score = -1.0
            return nodes

    apply_chain([Rescore()], nodes, "q")
    assert [n.score for n in nodes] == [0.9, 0.5]


def test_long_context_reorder_puts_best_at_the_ends():
    out = apply_chain([long_context_reorder()], _nodes(0.9, 0.8, 0.7, 0.6, 0.5), "q")
    scores = [n.score for n in out]
    assert len(scores) == 5 and {scores[0], scores[-1]} == {0.9, 0.8} and scores[2] == 0.5


def test_out_of_scope_rate():
    assert out_of_scope_rate(lambda q: [], ["a", "b"]) == 0.0
    assert out_of_scope_rate(lambda q: _nodes(0.5), ["a", "b"]) == 1.0

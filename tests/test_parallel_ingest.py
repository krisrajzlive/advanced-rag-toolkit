from llama_index.core.schema import Document

from src.ingestion.parallel import default_strategies, replicate


def test_replicate_makes_unique_ids_and_keeps_metadata():
    docs = [Document(text="a", id_="x", metadata={"m": 1}), Document(text="b", id_="y")]
    out = replicate(docs, 3)
    assert len(out) == 6 and len({d.doc_id for d in out}) == 6
    assert out[0].metadata == {"m": 1} and out[0].doc_id == "x#0"


def test_default_strategies_start_with_baseline_and_cover_each_lever():
    s = default_strategies(process_workers=3, async_workers=6)
    assert s[0].name.startswith("baseline") and not s[0].use_async
    assert any(x.embed_batch_size > 10 for x in s)
    assert any(x.use_async and x.embed_workers == 6 for x in s)
    assert any(x.process_workers == 3 for x in s)

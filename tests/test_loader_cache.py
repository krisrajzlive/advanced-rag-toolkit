from pathlib import Path

from src.loaders import cache


def test_cached_text_calls_producer_once_and_survives_failures(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"x" * 10)
    calls = []

    def produce():
        calls.append(1)
        return "hello transcript"

    assert cache.cached_text(media, "transcript", produce) == "hello transcript"
    assert cache.cached_text(media, "transcript", produce) == "hello transcript"
    assert len(calls) == 1  # second call served from disk

    media.write_bytes(b"y" * 20)  # file changed -> cache entry no longer applies
    assert cache.cached_text(media, "transcript", produce) == "hello transcript"
    assert len(calls) == 2


def test_failing_producer_returns_none_instead_of_raising(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"x")

    def boom():
        raise RuntimeError("402 Payment Required")

    assert cache.cached_text(media, "transcript", boom) is None
    assert not (tmp_path / "cache").exists()  # failures are not cached

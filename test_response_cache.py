from pathlib import Path

from mas.response_cache import ResponseCache


def test_response_cache_round_trip():
    path = Path(".mas_response_cache_test.jsonl")
    try:
        path.unlink(missing_ok=True)
        cache = ResponseCache(enabled=True, path=str(path), namespace="test")
        fingerprint = {"model": "unit-test", "tools": ["search_web"]}
        key = cache.make_key(agent_name="Planner", prompt="hello", fingerprint=fingerprint)

        assert cache.get(key) is None
        assert cache.put(
            key,
            agent_name="Planner",
            prompt="hello",
            fingerprint=fingerprint,
            text="FINAL_ANSWER: hi",
            tool_names=["search_web"],
        )

        hit = cache.get(key)
        assert hit is not None
        assert hit["text"] == "FINAL_ANSWER: hi"
        assert hit["tool_names"] == ["search_web"]

        reloaded = ResponseCache(enabled=True, path=str(path), namespace="test")
        assert reloaded.get(key)["text"] == "FINAL_ANSWER: hi"
    finally:
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    test_response_cache_round_trip()
    print("response cache tests passed")

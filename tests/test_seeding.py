from src.pipeline.caching import compute_cache_key
from src.pipeline.seeding import compute_seed


def test_seeding_and_cache_key_determinism():
    steps = [{"name": "jpeg", "params": {"quality": 60}}]
    seed1 = compute_seed(123, "img001", "recipeA", 0)
    seed2 = compute_seed(123, "img001", "recipeA", 0)
    assert seed1 == seed2

    key1 = compute_cache_key("/tmp/img001.jpg", steps, seed1, 0)
    key2 = compute_cache_key("/tmp/img001.jpg", steps, seed2, 0)
    assert key1 == key2

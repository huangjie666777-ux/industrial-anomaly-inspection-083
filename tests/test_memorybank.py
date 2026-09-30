import pytest
import torch

from inspection.memorybank import (
    build_memory_bank,
    heatmap_upper_bound,
    linear_quantile,
    nearest_distances,
)


def test_greedy_selection_deterministic_and_capped():
    descriptors = torch.randn(600, 32)
    bank_a = build_memory_bank(descriptors, max_items=256)
    bank_b = build_memory_bank(descriptors, max_items=256)
    assert bank_a.shape == (256, 32)
    assert torch.equal(bank_a, bank_b)


def test_greedy_includes_seed_and_distinct_points():
    points = torch.tensor([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
    bank = build_memory_bank(points, max_items=4)
    assert bank.shape[0] == 4
    assert torch.allclose(bank[0], points[0])


def test_duplicate_points_do_not_inflate_bank():
    points = torch.tensor([[0.0], [0.0], [0.0], [5.0]])
    bank = build_memory_bank(points, max_items=4)
    assert bank.shape[0] == 2


def test_nearest_distance_and_image_score():
    bank = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
    descriptors = torch.tensor([[0.0, 1.0], [3.0, 4.0]])
    distances = nearest_distances(descriptors, bank)
    assert torch.allclose(distances, torch.tensor([1.0, 0.0]), atol=1e-5)
    assert float(distances.max()) == 1.0


def test_linear_quantile_95():
    values = torch.arange(1.0, 101.0)
    assert linear_quantile(values, 95.0) == pytest.approx(95.05)


def test_heatmap_upper_bound_rules():
    scores = torch.tensor([0.1, 0.5, 0.2])
    assert heatmap_upper_bound(0.4, scores) == 0.8
    assert heatmap_upper_bound(0.0, scores) == 0.5
    assert heatmap_upper_bound(0.0, torch.zeros(3)) == 1e-6

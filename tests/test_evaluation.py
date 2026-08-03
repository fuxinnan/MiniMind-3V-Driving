import torch

from evaluate.control_accuracy import ControlAccuracyEvaluator
from evaluate.safety_metrics import SafetyEvaluator


def test_control_metrics_use_normalized_steering_scale():
    predicted = torch.tensor([[0.1, 0.2, 0.0, 2.0]])
    target = torch.tensor([[0.0, 0.2, 0.0, 2.0]])
    metrics = ControlAccuracyEvaluator(steering_max_degrees=30).evaluate(
        predicted, target
    )
    assert metrics.steering_mae == 3.0


def test_trajectory_metrics_are_not_claimed_without_trajectory():
    controls = torch.tensor([[0.0, 0.2, 0.0, 2.0]])
    metrics = SafetyEvaluator().evaluate(controls)
    assert "collision" not in metrics.available_metrics
    assert "braking" in metrics.available_metrics

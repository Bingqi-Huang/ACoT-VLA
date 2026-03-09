import numpy as np

from openpi.training import offline_eval as _offline_eval


def test_offline_eval_accumulator_reports_expected_metrics():
    accumulator = _offline_eval.OfflineEvalAccumulator(action_dim=8, horizon=3)
    accumulator.update(
        loss_per_example=np.asarray([1.0, 3.0]),
        predicted_actions=np.asarray(
            [
                [[0.0] * 7 + [0.9], [0.1] * 7 + [0.1], [0.2] * 7 + [0.8]],
                [[0.1] * 7 + [0.2], [0.2] * 7 + [0.7], [0.3] * 7 + [0.3]],
            ]
        ),
        target_actions=np.asarray(
            [
                [[0.0] * 7 + [1.0], [0.0] * 7 + [0.0], [0.0] * 7 + [1.0]],
                [[0.0] * 7 + [0.0], [0.0] * 7 + [1.0], [0.0] * 7 + [0.0]],
            ]
        ),
        task_names=np.asarray(["task_a", "task_b"]),
    )

    metrics = accumulator.finalize()

    assert metrics["overall_val_loss"] == 2.0
    assert metrics["per_task_val_loss"]["task_a"] == 1.0
    assert metrics["per_task_val_loss"]["task_b"] == 3.0
    assert len(metrics["per_action_dim_mae"]) == 8
    assert len(metrics["horizon_step_mae"]) == 3
    assert "overall_joint_mae" in metrics
    assert metrics["gripper"]["is_binary_target"] is True

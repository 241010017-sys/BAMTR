import math

import numpy as np
import pytest
import torch

from bamtr.calendar import environment_features, holiday_status
from bamtr.engine import teacher_forcing_ratio
from bamtr.metrics import EARTH_RADIUS_M, MetricAccumulator, distances_m, multitask_loss


def test_known_distance_and_micro_aggregation():
    degrees_per_meter = 180 / (math.pi * EARTH_RADIUS_M)
    truth = np.zeros((3, 2))
    prediction = np.array([[0, 0], [100 * degrees_per_meter, 0], [200 * degrees_per_meter, 0]])
    errors = distances_m(prediction, truth)
    np.testing.assert_allclose(errors, [0, 100, 200], atol=1e-7)
    metric = MetricAccumulator([50, 150], reference_latitude=0)
    metric.update(prediction[:1], truth[:1], [0.9], [1])
    metric.update(prediction[1:], truth[1:], [0.8, 0.1], [0, 0])
    result = metric.compute()
    assert result['mae_m'] == pytest.approx(100)
    assert result['rmse_m'] == pytest.approx(math.sqrt(50_000 / 3))
    assert result['recall_at_m'] == {'50': 1 / 3, '150': 2 / 3}
    assert result['stay_confusion'] == {'tp': 1, 'fp': 1, 'fn': 0, 'tn': 1}


def test_loss_excludes_nan_padding_and_uses_squared_vector_norm():
    prediction = {'coords': torch.tensor([[[1., 2.], [float('nan'), float('nan')]]], requires_grad=True),
                  'stay_logits': torch.tensor([[0., float('nan')]], requires_grad=True)}
    batch = {'target_coords': torch.tensor([[[0., 0.], [float('nan'), float('nan')]]]),
             'target_stay': torch.tensor([[1., float('nan')]]), 'target_mask': torch.tensor([[True, False]])}
    losses = multitask_loss(prediction, batch, 10)
    assert losses['coordinate_loss'].item() == 5
    assert losses['semantic_loss'].item() == pytest.approx(math.log(2))
    losses['loss'].backward()
    assert torch.isfinite(prediction['coords'].grad).all()
    assert torch.isfinite(prediction['stay_logits'].grad).all()
    assert prediction['coords'].grad[0, 1].abs().sum() == 0


def test_holiday_calendar_handles_makeup_workdays_and_unknown_years():
    assert holiday_status('2023-04-05') == 1
    assert holiday_status('2023-04-08') == 1
    assert holiday_status('2023-04-23') == 0  # Sunday, but an official working day.
    assert holiday_status('2024-04-05') == 2  # Explicitly outside bundled coverage.
    features = environment_features('2023-04-05', 12 * 3600)
    np.testing.assert_allclose(features, [.5, 2 / 6, 0, 1, 0])


def test_teacher_forcing_schedule_endpoints():
    assert teacher_forcing_ratio(0, 30, .8, .3) == .8
    assert teacher_forcing_ratio(29, 30, .8, .3) == pytest.approx(.3)
    assert teacher_forcing_ratio(0, 1, .8, .3) == .8

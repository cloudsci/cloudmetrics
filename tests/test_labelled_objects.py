"""
Try running all functions for labelled objects directly on masks
"""
import numpy as np
import pytest

from cloudmetrics.objects import label as label_objects
from cloudmetrics.objects import metrics as obj_metrics


@pytest.mark.parametrize("metric_name", obj_metrics.ALL_METRIC_FUNCTIONS.keys())
def test_all_object_metrics_with_random_mask(metric_name):
    N = 100
    mask = np.random.random((N, N)) > 0.8
    object_labels = label_objects(mask=mask)
    fn = obj_metrics.ALL_METRIC_FUNCTIONS[metric_name]
    fn(object_labels=object_labels)

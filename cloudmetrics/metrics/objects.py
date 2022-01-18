"""
Routines for evaluating cloud object metrics directly from cloud masks
"""
from ..objects import label as label_objects
from ..objects import metrics as obj_metrics
from ..utils import make_periodic_cloud_mask, print_object_labels


def _evaluate_metric(metric_name, cloud_mask, periodic_domain, object_connectivity=1):
    """
    Identify individual clouds in the cloud mask and return the number of
    objects
    """
    try:
        metric_function = getattr(obj_metrics, metric_name)
    except AttributeError:
        raise NotImplementedError(f"Object metric `{metric_name}` not implemented")
    if periodic_domain:
        cloud_mask = make_periodic_cloud_mask(
            field=cloud_mask, object_connectivity=object_connectivity
        )
        print_object_labels(cloud_mask)

    cloud_object_labels = label_objects(
        cloud_mask=cloud_mask, connectivity=object_connectivity
    )

    return metric_function(cloud_object_labels=cloud_object_labels)


def num_clouds(cloud_mask, periodic_domain, object_connectivity=1):
    """
    Identify individual clouds in the cloud mask and return the number of
    objects
    """
    return _evaluate_metric(
        metric_name="num_clouds",
        cloud_mask=cloud_mask,
        periodic_domain=periodic_domain,
        object_connectivity=object_connectivity,
    )


def max_length_scale(cloud_mask, periodic_domain, object_connectivity=1):
    """
    Identify individual clouds objects in the cloud mask and calculate the maximum length-scale across all objects
    """
    return _evaluate_metric(
        metric_name="max_length_scale",
        cloud_mask=cloud_mask,
        periodic_domain=periodic_domain,
        object_connectivity=object_connectivity,
    )


def mean_length_scale(cloud_mask, periodic_domain, object_connectivity=1):
    """
    Identify individual clouds objects in the cloud mask and calculate the mean
    length-scale (evaluated as the square-root of the mean areas) across all objects
    """
    return _evaluate_metric(
        metric_name="mean_length_scale",
        cloud_mask=cloud_mask,
        periodic_domain=periodic_domain,
        object_connectivity=object_connectivity,
    )


def mean_perimeter_length(cloud_mask, periodic_domain, object_connectivity=1):
    """
    Identify individual clouds objects in the cloud mask and calculate the mean
    perimeter length (in grid units) across all objects
    """
    return _evaluate_metric(
        metric_name="mean_perimeter_length",
        cloud_mask=cloud_mask,
        periodic_domain=periodic_domain,
        object_connectivity=object_connectivity,
    )


def mean_eccentricity(cloud_mask, periodic_domain, object_connectivity=1):
    """
    Identify individual clouds objects in the cloud mask and calculate the
    eccentricity across all objects
    """
    return _evaluate_metric(
        metric_name="mean_eccentricity",
        cloud_mask=cloud_mask,
        periodic_domain=periodic_domain,
        object_connectivity=object_connectivity,
    )

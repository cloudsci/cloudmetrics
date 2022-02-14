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

    return metric_function(
        object_labels=cloud_object_labels, periodic_domain=periodic_domain
    )


def _make_mask_function_name(metric_name):
    """
    Generate a name for the metric function that applies directly to a mask

    The naming convention for object functions that apply directly to masks is
        `{op}_{measure}` -> `{op}_cloud_{measure}`,
    e.g.
        `num_objects` -> `num_objects`
        `mean_perimeter_length` -> `mean_object_perimeter_length`
        `cop` -> `cop_objects`
    """
    if "objects" in metric_name:
        return metric_name

    op, *measure_parts = metric_name.split("_")
    if len(measure_parts) == 0:
        return f"{op}_objects"
    else:
        return "_".join([op, "object"] + measure_parts)


_OBJECT_FUNCTION_TEMPLATE = """
def {function_name}(cloud_mask, periodic_domain, object_connectivity=1):
    '''{docstring}'''
    return _evaluate_metric(
        metric_name="{metric_name}",
        cloud_mask=cloud_mask,
        periodic_domain=periodic_domain,
        object_connectivity=object_connectivity,
    )
"""


def _make_mask_function_strings():
    for (metric_name, fn) in obj_metrics.ALL_METRIC_FUNCTIONS.items():
        metric_docstring = fn.__doc__.strip()
        function_name = _make_mask_function_name(metric_name=metric_name)
        metric_docstring = metric_docstring[0].lower() + metric_docstring[1:]
        docstring = f"Identify individual objects in the mask and {metric_docstring}"
        mask_function_str = _OBJECT_FUNCTION_TEMPLATE.format(
            metric_name=metric_name, docstring=docstring, function_name=function_name
        )
        yield mask_function_str


for _mask_function_string in _make_mask_function_strings():
    exec(_mask_function_string)

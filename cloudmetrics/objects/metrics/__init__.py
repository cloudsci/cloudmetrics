import inspect
import sys

from .geometry import (  # noqa
    max_length_scale,
    mean_eccentricity,
    mean_length_scale,
    mean_perimeter_length,
)
from .num_objects import num_objects  # noqa
from .scai import scai  # noqa


def _find_labelled_objects_functions():
    """
    Look through the functions available in this module and return all the ones
    that look like they operate on labelled objects
    """

    def _takes_object_labels_kwarg(fn):
        fn_sig = inspect.signature(fn)
        return "object_labels" in fn_sig.parameters and len(fn_sig.parameters) == 1

    fns = [
        (fn_name, fn)
        for (fn_name, fn) in inspect.getmembers(
            sys.modules[__name__], inspect.isfunction
        )
        if not fn_name.startswith("_") and _takes_object_labels_kwarg(fn)
    ]

    return dict(fns)


ALL_METRIC_FUNCTIONS = _find_labelled_objects_functions()

import inspect
import sys

from .cop import cop  # noqa
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

    def _num_args_without_default_value(fn_sig):
        return len(
            [
                param
                for param in fn_sig.parameters.values()
                if param.default is inspect._empty
            ]
        )

    def _takes_object_labels_kwarg(fn):
        fn_sig = inspect.signature(fn)
        return (
            "object_labels" in fn_sig.parameters
            and _num_args_without_default_value(fn_sig) == 1
        )

    fns = [
        (fn_name, fn)
        for (fn_name, fn) in inspect.getmembers(
            sys.modules[__name__], inspect.isfunction
        )
        if not fn_name.startswith("_") and _takes_object_labels_kwarg(fn)
    ]

    return dict(fns)


ALL_METRIC_FUNCTIONS = _find_labelled_objects_functions()

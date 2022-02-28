import inspect
import sys

from .cloud_fraction import cloud_fraction  # noqa
from .fractal_dimension import fractal_dimension  # noqa
from .iorg import iorg  # noqa
from .objects import *  # noqa
from .open_sky import open_sky  # noqa
from .orientation import orientation  # noqa


def _find_mask_functions():
    """
    Look through the functions available in this module and return all the ones
    that look like they operate on object masks
    """

    def _takes_object_mask_kwarg(fn):
        fn_sig = inspect.signature(fn)
        return "mask" in fn_sig.parameters

    fns = [
        (fn_name, fn)
        for (fn_name, fn) in inspect.getmembers(
            sys.modules[__name__], inspect.isfunction
        )
        if not fn_name.startswith("_") and _takes_object_mask_kwarg(fn)
    ]

    return dict(fns)


ALL_METRIC_FUNCTIONS = _find_mask_functions()

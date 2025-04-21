import pdb
import torch
from typing import (
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    TypeVar,
    Sequence,
    Tuple,
)

import torch.nn as nn
import inspect

def get_module_forward_input_names(module: nn.Module):
    
    params = inspect.signature(module.forward).parameters
    param_names = [k for k, v in params.items() if not str(v).startswith("*")]
    print('stop for inepecting names')
    #pdb.set_trace()
    return param_names



def maybe_len(obj) -> Optional[int]:
    try:
        return len(obj)
    except (NotImplementedError, AttributeError):
        return None


def copy_parameters(
    net_source: torch.nn.Module,
    net_dest: torch.nn.Module,
    strict: Optional[bool] = True,
) -> None:
    """
    Copies parameters from one network to another.
    Parameters
    ----------
    net_source
        Input network.
    net_dest
        Output network.
    strict:
        whether to strictly enforce that the keys
        in :attr:`state_dict` match the keys returned by this module's
        :meth:`~torch.nn.Module.state_dict` function. Default: ``True``
    """

    net_dest.load_state_dict(net_source.state_dict(), strict=strict)


from typing import List, Optional

import numpy as np
from pandas.tseries.frequencies import to_offset

from gluonts.time_feature import norm_freq_str


def _make_lags(middle: int, delta: int) -> np.ndarray:
    """
    Create a set of lags around a middle point including +/- delta.
    """
    print('the actual working code of calculating lags ')
    print('\n')
    #pdb.set_trace()
    return np.arange(middle - delta, middle + delta + 1).tolist()


def get_lags_for_frequency(
    freq_str: str, lag_ub: int = 1200, num_lags: Optional[int] = None
) -> List[int]:
    """
    Generates a list of lags that that are appropriate for the given frequency
    string.
    By default all frequencies have the following lags: [1, 2, 3, 4, 5, 6, 7].
    Remaining lags correspond to the same `season` (+/- `delta`) in previous
    `k` cycles. Here `delta` and `k` are chosen according to the existing code.
    Parameters
    ----------
    freq_str
        Frequency string of the form [multiple][granularity] such as "12H",
        "5min", "1D" etc.
    lag_ub
        The maximum value for a lag.
    num_lags
        Maximum number of lags; by default all generated lags are returned
    """

    # Lags are target values at the same `season` (+/- delta) but in the
    # previous cycle.
    def _make_lags_for_second(multiple, num_cycles=3):
        # We use previous ``num_cycles`` hours to generate lags
        return [
            _make_lags(k * 60 // multiple, 2) for k in range(1, num_cycles + 1)
        ]

    def _make_lags_for_minute(multiple, num_cycles=3):
        print('make lags for minute')
        print('\n')
        #pdb.set_trace()
        # We use previous ``num_cycles`` hours to generate lags
        return [
            _make_lags(k * 60 // multiple, 2) for k in range(1, num_cycles + 1)
        ]

    def _make_lags_for_hour(multiple, num_cycles=7):
        print('make lags for hour')
        print('\n')
        #pdb.set_trace()
        # We use previous ``num_cycles`` days to generate lags
        return [
            _make_lags(k * 24 // multiple, 1) for k in range(1, num_cycles + 1)
        ]

    def _make_lags_for_day(multiple, num_cycles=4):
        print('make lags for day')
        print('\n')
        #pdb.set_trace()
        # We use previous ``num_cycles`` weeks to generate lags
        # We use the last month (in addition to 4 weeks) to generate lag.
        return [
            _make_lags(k * 7 // multiple, 1) for k in range(1, num_cycles + 1)
        ] + [_make_lags(30 // multiple, 1)]

    def _make_lags_for_week(multiple, num_cycles=3):
        print('make lags for week')
        print('\n')
        #pdb.set_trace()
        # We use previous ``num_cycles`` years to generate lags
        # Additionally, we use previous 4, 8, 12 weeks
        return [
            _make_lags(k * 52 // multiple, 1) for k in range(1, num_cycles + 1)
        ] + [[4 // multiple, 8 // multiple, 12 // multiple]]

    def _make_lags_for_month(multiple, num_cycles=3):
        # We use previous ``num_cycles`` years to generate lags
        return [
            _make_lags(k * 12 // multiple, 1) for k in range(1, num_cycles + 1)
        ]

    # multiple, granularity = get_granularity(freq_str)
    offset = to_offset(freq_str)
    # normalize offset name, so that both `W` and `W-SUN` refer to `W`
    offset_name = norm_freq_str(offset.name)

    if offset_name == "A":
        lags = []
    elif offset_name == "Q":
        assert (
            offset.n == 1
        ), "Only multiple 1 is supported for quarterly. Use x month instead."
        lags = _make_lags_for_month(offset.n * 3.0)
    elif offset_name == "M":
        lags = _make_lags_for_month(offset.n)
    elif offset_name == "W":
        lags = _make_lags_for_week(offset.n)
    elif offset_name == "D":
        lags = _make_lags_for_day(offset.n) + _make_lags_for_week(
            offset.n / 7.0
        )
    elif offset_name == "B":
        # todo find good lags for business day
        lags = []
    elif offset_name == "H":
        lags = (
            _make_lags_for_hour(offset.n)
            + _make_lags_for_day(offset.n / 24)
            + _make_lags_for_week(offset.n / (24 * 7))
        )
    # minutes
    elif offset_name == "T":

        print('how the indices are computed?')
        print('\n')

        #pdb.set_trace()
        lags = (
            _make_lags_for_minute(offset.n)
            + _make_lags_for_hour(offset.n / 60)
            + _make_lags_for_day(offset.n / (60 * 24))
            + _make_lags_for_week(offset.n / (60 * 24 * 7))
        )
    # second
    elif offset_name == "S":
        lags = (
            _make_lags_for_second(offset.n)
            + _make_lags_for_minute(offset.n / 60)
            + _make_lags_for_hour(offset.n / (60 * 60))
        )
    else:
        raise Exception("invalid frequency")

    
    print('\n')
    print("what's this")
    #pdb.set_trace()
    # flatten lags list and filter
    lags = [
        int(lag) for sub_list in lags for lag in sub_list if 7 < lag <= lag_ub
    ]
    print('\n')
    print(' how to define lags??? confusing??')
    #pdb.set_trace()
    lags = [1, 2, 3, 4, 5, 6, 7] + sorted(list(set(lags)))
    

    return lags[:num_lags]

import torch.nn as nn


class LambdaLayer(nn.Module):
    def __init__(self, function):
        super().__init__()
        self._func = function

    def forward(self, x, *args):
        return self._func(x, *args)
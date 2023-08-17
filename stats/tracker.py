from typing import Any, Dict, List, Optional

import torch

from .dict_utils import get_recursive, set_recursive
from .metric import Metric


class StatsTracker:
    """
    For tracking stats

    Stores the stats in a list for each metric. The metrics can be given as nested keys
    by separating them with dots, for example: accuracy.point.

    The `keys` define the hierarchy of the stats, and in order to add a stat for new
    timestep, a dictionary with the same hierarchy is given with a single value, which
    is then added to the list of the respective keys.

    Example:

    keys = ["loss", "accuracy.point", "accuracy.box"]
    Gives the following stats in the tracker:
        StatsTracker({"loss": [], "accuracy": {"point": [], "box": []"}})

    The first result is then added to the tracker:
        result = {"loss": 0.8, "accuracy": {"point": 0.91, "box": 0.77}}
        stats_tracker.append(result)

    Which means the stats in the tracker are now:
        StatsTracker({"loss": [0.8], "accuracy": {"point": [0.91], "box": [0.77]"}})
    """

    # Keys of the stats
    # May be nested, separated by dots, for example: accuracy.point
    keys = [
        "loss",
        "accuracy",
        "iou",
    ]

    def __init__(self, keys: Optional[List[str]] = None, extend: bool = True):
        """
        Args:
            keys (List[str], optional): Keys to use for the stats. If not
                specified, the default keys are used.
            extend (bool): Whether to extend the existing keys with the
                given keys instead of replacing them completely.
                [Default: True]
        """
        super().__init__()
        self.stats: Dict = {}
        if keys is not None:
            if extend:
                # Make sure the keys are unique otherwise that would have negative
                # effects when appending.
                self.keys = list(set(self.keys + keys))
            else:
                self.keys = keys
        for key in self.keys:
            set_recursive(self.stats, key, [])

    @classmethod
    def with_metrics(
        cls, metrics: List[Metric], extend: bool = False
    ) -> "StatsTracker":
        """
        Creates the StatsTracker with the appropriate keys for the given metrics.

        Args:
            metrics (List[Metric]): List of metrics that should be tracked.
            extend (bool): Whether to extend the existing keys with the
                given keys instead of replacing them completely.
                [Default: False]

        Returns:
            stats_tracker (StatsTracker): The stats tracker for the given metrics.
        """
        stats_tracker = cls(keys=[m.key for m in metrics], extend=extend)
        return stats_tracker

    @classmethod
    def from_list(
        cls, stats_list: List[Dict], metrics: Optional[List[Metric]] = None
    ) -> "StatsTracker":
        """
        Creates the StatsTracker from a list of stats.

        Args:
            stats_list (List[Dict]): List of stats
            metrics (List[Metric], optional): List of metrics that should be tracked.
                Can be specified in order to track different stats.

        Returns:
            stats_tracker (StatsTracker): The stats tracker initialised with the given
                stats.
        """
        stats_tracker = cls() if metrics is None else cls.with_metrics(metrics)
        for stat in stats_list:
            stats_tracker.append(stat)
        return stats_tracker

    @classmethod
    def from_dict(
        cls, stats_dict: Dict, metrics: Optional[List[Metric]] = None
    ) -> "StatsTracker":
        """
        Creates the StatsTracker from a dictionary of stats.

        Args:
            stats_list (Dict): Dictionary of stats
            metrics (List[Metric], optional): List of metrics that should be tracked.
                Can be specified in order to track different stats.

        Returns:
            stats_tracker (StatsTracker): The stats tracker initialised with the given
                stats.
        """
        stats_tracker = cls() if metrics is None else cls.with_metrics(metrics)
        for key in stats_tracker.keys:
            stat = get_recursive(stats_dict, key)
            if stat is not None:
                assert isinstance(stat, list), (
                    ".from_dict() expects values for the keys to be lists "
                    f"- {repr(key)} is not a list"
                )
                set_recursive(stats_tracker.stats, key, stat)
        return stats_tracker

    def get(self, key: Optional[str] = None) -> Any:
        """
        Get the value for the nested key out of the tracker. If no key is given, it
        returns the whole tracked values as a dictionary,

        Args:
            key (str, optional): Key to lookup. If not given returns the whole
                dictionary.

        Returns:
            value (Any): The value of the given key.
        """
        if key is None:
            return self.stats
        return get_recursive(self.stats, key)

    def append(self, other: Dict):
        """
        Appends a set of stats to the tracker. The stats are given as a dictionary with
        the same hierarchy as the tracker, but with a single value.

        Args:
            other (Dict): Dictionary with the stats to be added.
        """
        for key in self.keys:
            other_value = get_recursive(other, key)
            if other_value is not None:
                stat = self.get(key)
                stat.append(other_value)

    def mean(self) -> Dict:
        """
        Calculates the mean values for all tracked stats across all time steps.
        Assumes that all values can be converted to a float.

        Returns:
            out (Dict): Dictionary with the mean values for all tracked stats.
        """
        out: Dict = {}
        for key in self.keys:
            stat = self.get(key)
            mean_value = (
                torch.mean(torch.tensor(stat, dtype=torch.float)).item()
                if len(stat) > 0
                else None
            )
            set_recursive(out, key, mean_value)
        return out

    def last(self) -> Dict:
        """
        Gives the last value for each tracked stat.

        Returns:
            out (Dict): Dictionary with the last value for each tracked stat.
        """
        out: Dict = {}
        for key in self.keys:
            stat = self.get(key)
            # Set the value to the last one if there is one, otherwise to
            # none
            last_value = stat[-1] if len(stat) > 0 else None
            set_recursive(out, key, last_value)
        return out

    def __repr__(self):
        return f"Stats({repr(self.stats)})"

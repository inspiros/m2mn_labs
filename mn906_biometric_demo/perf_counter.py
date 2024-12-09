from typing import Any, Optional

import time

__all__ = ['AverageCounter', 'PerfCounter']


class AverageCounter:
    r'''A class for computing running average.'''

    def __init__(self, momentum: Optional[float] = .9) -> None:
        if momentum is not None and (momentum < 0 or momentum > 1):
            raise ValueError('momentum must be in range [0, 1]')
        self.running_mean: Any = 0
        self.count: int = 0
        self.momentum = momentum

    def reset(self) -> None:
        self.running_mean = 0
        self.count = 0

    def update(self, x: Any) -> None:
        if self.momentum is not None:
            self.running_mean = \
                self.running_mean * self.momentum + x * (1 - self.momentum)
        else:
            self.running_mean = (self.running_mean * self.count + x) / (self.count + 1)
            self.count += 1

    def update_(self, x: Any) -> Any:
        self.update(x)
        return self.running_mean

    def __repr__(self) -> str:
        res = f'{self.__class__.__name__}('
        if self.momentum is not None:
            res += f'momentum={self.momentum:.03f}'
        return res + ')'


class PerfCounter(AverageCounter):
    r'''Performance Measuring Context Manager.'''

    def __init__(self, momentum: Optional[float] = .9) -> None:
        super().__init__(momentum)
        self.start = self.end = None

    @property
    def running_elapsed_time_ns(self) -> int:
        return self.running_mean

    @running_elapsed_time_ns.setter
    def running_elapsed_time_ns(self, val: int) -> None:
        self.running_mean = val

    @property
    def last_elapsed_time_ns(self) -> int:
        return self.end - self.start

    @property
    def last_elapsed_time(self) -> float:
        return (self.end - self.start) / 1e9

    @property
    def running_elapsed_time(self) -> float:
        return self.running_elapsed_time_ns / 1e9

    @property
    def fps(self) -> float:
        return 1e9 / self.running_elapsed_time_ns if self.running_elapsed_time_ns else float('inf')

    def reset(self) -> None:
        super().reset()
        self.start = self.end = None

    def __enter__(self):
        self.start = time.perf_counter_ns()

    def __exit__(self, type, value, traceback):
        self.end = time.perf_counter_ns()
        self.update(self.last_elapsed_time_ns)

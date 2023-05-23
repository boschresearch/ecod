# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import logging
import time
from contextlib import contextmanager


class ProgressLogger:
    def __init__(self, iterable_or_len=None, name="", every_n_percent=10.0):
        try:
            self.total_len = int(iterable_or_len)
            self.iterable = range(self.total_len)
        except TypeError:
            self.iterable = iterable_or_len
            self.total_len = len(self.iterable)
        self.name = name
        self.every_n_percent = every_n_percent
        self.every_n_steps = every_n_percent / 100.0 * self.total_len
        self.n = None
        self.next_print = None
        self.last_print_n = None
        self.t_start = None
        self.last_time = None
        self.did_print = False
        self.logger = logging.getLogger(f"{name.upper()}")
        self.logger.setLevel("INFO")

    def __iter__(self):
        self.start()
        for obj in self.iterable:
            self.update(1)
            yield obj

    def start(self):
        self.n = 1
        self.next_print = self.every_n_steps
        self.last_print_n = 1
        self.did_print = False
        self.t_start = time.time()
        self.last_time = self.t_start
        return self

    def update(self, n=1):
        if self.n >= self.next_print or self.n == self.total_len:
            self.next_print += self.every_n_steps
            # while self.n >= self.next_print:
            #    self.next_print += self.every_n_steps
            perc, tot, t_it, t_tot, t_now, t_fin = self.get_print_vars()
            if t_fin > 3600:
                t_fin /= 3600
                hs = "h"
            else:
                hs = "s"
            self.logger.info(f"Reached {perc} ({tot}), t_it={t_it:.2f}s, t_tot={t_tot:.2f}s, t_fin={t_fin:.2f}{hs}")
            self.last_time = t_now
            self.did_print = True
            self.last_print_n = 1
        else:
            perc, tot, t_it, t_tot, t_now = [None] * 5
            self.did_print = False
        self.n += n
        self.last_print_n += n
        return self.did_print, perc, tot, t_it, t_tot, t_now

    def get_print_vars(self):
        perc = f"{100. * (self.n) / self.total_len:.1f}%"
        tot = f"{self.n}/{self.total_len}"
        t_now = time.time()
        t_it = t_now - self.last_time
        t_tot = t_now - self.t_start
        # estimate from last time not from start, because often first step takes longer, leading to skewed estimate
        # t_fin = t_tot / self.n * self.total_len - t_tot
        t_fin = t_it * (self.total_len - self.n) / self.last_print_n
        return perc, tot, t_it, t_tot, t_now, t_fin

    def __len__(self):
        return self.total_len


@contextmanager
def optional(condition, context_manager):
    if condition:
        with context_manager:
            yield
    else:
        yield

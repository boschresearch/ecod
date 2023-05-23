# Copyright (c) 2022 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from abc import ABC, abstractclassmethod, abstractproperty
import numpy as np
import logging

import torch


class ThresholdLogger:
    def __init__(self, log_step=1.0, name="THRESH", unit="GB"):
        self.log_step = log_step
        self.unit = unit
        self._cur_log_idx = 1
        self.logger = logging.getLogger(name)
        self.logger.setLevel("INFO")

    def __call__(self, value):
        low = self.log_step * (self._cur_log_idx - 1)
        high = self.log_step * self._cur_log_idx
        if value > high:
            self.logger.info(f"Increased: {value:.2g} {self.unit}")
            self._cur_log_idx += int(value / high)
        elif value < low:
            self.logger.info(f"Decreased: {value:.2g} {self.unit}")
            if value == 0:
                self._cur_log_idx = 1
            else:
                self._cur_log_idx -= int(low / value)


class AbstractStackedArray(ABC):
    def __init__(self, accum_len=10000, concat_dim=0, name=""):
        self.accum_len = accum_len
        self.concat_dim = concat_dim
        self.name = name
        self.stack = []
        self.is_numpy = None
        self.cur_accum_idx = 0
        self.logger = ThresholdLogger(name=name)

    def __call__(self, array):
        raise NotImplementedError()

    def __getitem__(self, idx):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def accumulate_stack(self):
        raise NotImplementedError()

    def check_numpy(self, array_type):
        if self.is_numpy is None:
            self.is_numpy = array_type == np.ndarray
        if self.is_numpy and array_type != np.ndarray:
            raise ValueError(
                f"Provided array has to be numpy but is of type {array_type}"
            )
        elif not self.is_numpy and array_type != torch.Tensor:
            raise ValueError(
                f"Provided array has to be torch but is of type {array_type}"
            )

    def delete(self, idx):
        raise NotImplementedError()

    def log_size(self):
        self.logger(self.array.nbytes / 1024**3)

    @abstractclassmethod
    def from_array(cls, *args, **kwargs):
        raise NotImplementedError()

    @abstractproperty
    def array(self):
        pass


class StackedArrayNEL(AbstractStackedArray):
    """StackedArray No Equal Lengths"""

    def __init__(self, accum_len=10000, concat_dim=0, name=""):
        super().__init__(accum_len, concat_dim, name)
        self.cum_lens = [0]

    def __call__(self, array_list):
        if type(array_list) != list:
            array_list = [array_list]
        for array in array_list:
            self.check_numpy(type(array))
            self.stack.append(array)
            self.cum_lens.append(self.cum_lens[-1] + array.shape[self.concat_dim])
            self.cur_accum_idx += 1
            if self.cur_accum_idx == self.accum_len:
                self.accumulate_stack()
                self.cur_accum_idx = 0

    def __getitem__(self, idx):
        if self.concat_dim != 0:
            raise NotImplementedError(
                "Have not implemented readout for concat_dim != 0"
            )
        if len(self.stack) != 1:
            self.accumulate_stack()
        start, stop = self.cum_lens[idx], self.cum_lens[idx + 1]
        return self.stack[0][start:stop]

    def __len__(self):
        return len(self.cum_lens) - 1

    def accumulate_stack(self):
        if len(self.stack) > 1:
            if self.is_numpy:
                self.stack = [np.concatenate(self.stack, self.concat_dim)]
                self.log_size()
            else:
                self.stack = [torch.cat(self.stack, self.concat_dim)]

    def delete(self, idx):
        if not self.is_numpy:
            raise NotImplementedError("Deleting is only implemented for numpy")
        self.accumulate_stack()
        start, stop = self.cum_lens[idx], self.cum_lens[idx + 1]
        n_eles = stop - start
        self.cum_lens = np.delete(self.cum_lens, idx, axis=0)
        self.cum_lens[idx:] -= n_eles
        delete_idxs = np.arange(start, stop, 1)
        self.stack[0] = np.delete(self.stack[0], delete_idxs, axis=self.concat_dim)

    def delete_from(self, idx):
        self.accumulate_stack()
        stop = self.cum_lens[idx + 1]
        self.cum_lens = self.cum_lens[: idx + 1]
        self.stack[0] = self.stack[0][:stop]
        return self

    def delete_range(self, start_idx, stop_idx):
        self.accumulate_stack()
        stop = self.cum_lens[stop_idx + 1]
        self.cum_lens = self.cum_lens[: stop_idx + 1]
        self.stack[0] = self.stack[0][:stop]
        start = self.cum_lens[start_idx]
        self.cum_lens = [cc - start for cc in self.cum_lens[start_idx:]]
        self.stack[0] = self.stack[0][start:]
        return self

    @property
    def array(self):
        self.accumulate_stack()
        return self.stack[0]

    @classmethod
    def from_array(cls, array, cum_lens, accum_len=10000, concat_dim=0, name=""):
        obj = cls(accum_len=accum_len, concat_dim=concat_dim, name=name)
        obj.stack = [array]
        obj.cum_lens = cum_lens
        obj.log_size()
        return obj

    def zero_elements_mask(self):
        return ~np.insert(np.diff(self.cum_lens) == 0, 0, False)

    def remove_zeros_with_mask(self, mask):
        self.cum_lens = list(np.array(self.cum_lens)[mask])

    def insert(self, idx, arr):
        self.accumulate_stack()
        insert_idx = self.cum_lens[idx + 1]
        self.stack[0] = np.insert(self.stack[0], insert_idx, arr, axis=self.concat_dim)
        self.cum_lens[idx + 1 :] = [ss + len(arr) for ss in self.cum_lens[idx + 1 :]]


class StackedArrayEL(AbstractStackedArray):
    """StackedArray with Equal Lengths"""

    def __init__(self, accum_len=10000, concat_dim=0, name=""):
        super().__init__(accum_len, concat_dim, name)
        self.cur_len = 0

    def __call__(self, array, accumulate_as_list=False):
        if not accumulate_as_list:
            array = [array]
        for arr in array:
            self.check_numpy(type(arr))
            self.stack.append(arr)
            self.cur_len += 1
            self.cur_accum_idx += 1
            if self.cur_accum_idx == self.accum_len:
                self.accumulate_stack()
                self.cur_accum_idx = 0

    def __getitem__(self, idx):
        if self.concat_dim != 0:
            raise NotImplementedError(
                "Have not implemented readout for concat_dim != 0"
            )
        if len(self.stack) != 1:
            self.accumulate_stack()
        return self.stack[0][idx]

    def __len__(self):
        return self.cur_len

    def accumulate_stack(self):
        if len(self.stack) > 1:
            if self.is_numpy:
                self.stack = [np.stack(self.stack, self.concat_dim)]
            else:
                self.stack = [torch.stack(self.stack, self.concat_dim)]

    def delete(self, idx):
        if not self.is_numpy:
            raise NotImplementedError("Deleting is only implemented for numpy")
        self.accumulate_stack()
        self.stack[0] = np.delete(self.stack[0], idx, axis=self.concat_dim)
        self.cur_len -= 1

    def delete_from(self, idx):
        self.accumulate_stack()
        self.stack[0] = self.stack[0][: idx + 1]
        self.cur_len = len(self.stack[0])
        return self

    def delete_range(self, start_idx, stop_idx):
        self.accumulate_stack()
        self.stack[0] = self.stack[0][start_idx : stop_idx + 1]
        self.cur_len = len(self.stack[0])
        return self

    @property
    def array(self):
        self.accumulate_stack()
        return self.stack[0]

    @classmethod
    def from_array(cls, array, cur_len, accum_len=10000, concat_dim=0, name=""):
        obj = cls(accum_len=accum_len, concat_dim=concat_dim, name=name)
        obj.stack = [array]
        obj.cur_len = cur_len
        obj.log_size()
        return obj

    def insert(self, idx, arr):
        raise RuntimeError(
            "This cannot be implemented because all elements have the same shape, "
            "inserting would change the shape of that element"
        )


class PreallocArrayNEL(AbstractStackedArray):
    """Preallocated array No Equal Lengths"""

    def __init__(self, accum_len=int(1e6), concat_dim=0, name=""):
        super().__init__(accum_len, concat_dim, name)
        self.cum_lens = [0]
        self.shape = None
        self.array = None
        self.dtype = None

    def __call__(self, array_list):
        if type(array_list) != list:
            array_list = [array_list]
        if self.shape is None:
            self.init_array(array_list[0])
        for array in array_list:
            n_eles = array.shape[self.concat_dim]
            self.cum_lens.append(self.cum_lens[-1] + n_eles)
            while self.cum_lens[-1] >= len(self.array):
                self.array = np.concatenate(
                    [self.array, np.empty(self.shape, dtype=self.dtype)]
                )
                self.log_size()
            start, stop = self.cum_lens[-2], self.cum_lens[-1]
            self.array[start:stop] = array

    def init_array(self, array):
        self.check_numpy(type(array))
        self.shape = list(array.shape)
        self.dtype = array.dtype
        self.shape[self.concat_dim] = self.accum_len
        if self.is_numpy:
            self.array = np.empty(self.shape, dtype=self.dtype)
        else:
            self.array = torch.empty(self.shape, dtype=self.dtype)
        self.log_size()

    def __getitem__(self, idx):
        if self.concat_dim != 0:
            raise NotImplementedError(
                "Have not implemented readout for concat_dim != 0"
            )
        start, stop = self.cum_lens[idx], self.cum_lens[idx + 1]
        return self.array[start:stop]

    def __len__(self):
        return len(self.cum_lens) - 1

    def accumulate_stack(self):
        self.array = self.array[: self.cum_lens[-1]]
        self.log_size()

    def delete(self, idx):
        if not self.is_numpy:
            raise NotImplementedError("Deleting is only implemented for numpy")
        start, stop = self.cum_lens[idx], self.cum_lens[idx + 1]
        n_eles = stop - start
        self.cum_lens = np.delete(self.cum_lens, idx, axis=0)
        self.cum_lens[idx:] -= n_eles
        delete_idxs = np.arange(start, stop, 1)
        self.array = np.delete(self.array, delete_idxs, axis=self.concat_dim)

    def delete_from(self, idx):
        self.accumulate_stack()
        stop = self.cum_lens[idx + 1]
        self.cum_lens = self.cum_lens[: idx + 1]
        self.stack[0] = self.stack[0][:stop]
        return self

    def delete_range(self, start_idx, stop_idx):
        self.accumulate_stack()
        stop = self.cum_lens[stop_idx + 1]
        self.cum_lens = self.cum_lens[: stop_idx + 1]
        self.stack[0] = self.stack[0][:stop]
        start = self.cum_lens[start_idx]
        self.cum_lens = [cc - start for cc in self.cum_lens[start_idx:]]
        self.stack[0] = self.stack[0][start:]
        return self

    @classmethod
    def from_array(cls, array, cum_lens, accum_len=int(1e6), concat_dim=0, name=""):
        obj = cls(accum_len=accum_len, concat_dim=concat_dim, name=name)
        obj.array = array
        obj.cum_lens = cum_lens
        obj.log_size()
        return obj


class StackedArrayCat(AbstractStackedArray):
    """StackedArray where all inputs are concatenated"""

    def __init__(self, accum_len=10000, concat_dim=0, name=""):
        super().__init__(accum_len, concat_dim, name)

    def __call__(self, array):
        self.check_numpy(type(array))
        self.stack.append(array)
        self.cur_accum_idx += 1
        if self.cur_accum_idx == self.accum_len:
            self.accumulate_stack()
            self.cur_accum_idx = 0

    def __getitem__(self, idx):
        if self.concat_dim != 0:
            raise NotImplementedError(
                "Have not implemented readout for concat_dim != 0"
            )
        if len(self.stack) != 1:
            self.accumulate_stack()
        return self.stack[0][idx]

    def __len__(self):
        self.accumulate_stack()
        return len(self.array)

    def accumulate_stack(self):
        if len(self.stack) > 1:
            if self.is_numpy:
                self.stack = [np.concatenate(self.stack, axis=self.concat_dim)]
            else:
                self.stack = [torch.cat(self.stack, axis=self.concat_dim)]

    def delete(self, idx):
        if not self.is_numpy:
            raise NotImplementedError("Deleting is only implemented for numpy")
        self.accumulate_stack()
        self.stack[0] = np.delete(self.stack[0], idx, axis=self.concat_dim)

    def delete_from(self, idx):
        self.accumulate_stack()
        self.stack[0] = self.stack[0][: idx + 1]
        return self

    def delete_range(self, start_idx, stop_idx):
        self.accumulate_stack()
        self.stack[0] = self.stack[0][start_idx : stop_idx + 1]
        return self

    @property
    def array(self):
        self.accumulate_stack()
        return self.stack[0]

    @classmethod
    def from_array(cls, array, accum_len=10000, concat_dim=0, name=""):
        obj = cls(accum_len=accum_len, concat_dim=concat_dim, name=name)
        obj.stack = [array]
        obj.log_size()
        return obj

    def insert(self, idx, arr):
        raise RuntimeError(
            "This cannot be implemented because all elements have the same shape, "
            "inserting would change the shape of that element"
        )


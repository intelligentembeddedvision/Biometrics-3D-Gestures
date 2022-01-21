import numpy as np
import array
from copy import deepcopy
import h5py
from math import ceil


def load_bin(file_name, data_format=None, padding=True):
    with open(file_name, 'rb') as f:
        bytes_array = f.read()

    double_array = array.array('d', bytes_array)
    np_array = np.array(double_array)

    if data_format is None:
        return np_array
    # matlab saves data columnwise, numpy does rowwise,
    # we must transpose the rows and columns
    # for a multidimensional array, we must transpose the last two dimensions
    # we must reshape the data with the last two dimensions swapped so the
    # matrix has the right shape after transposing

    data_format = deepcopy(data_format)
    cup = data_format[-2]
    data_format[-2] = data_format[-1]
    data_format[-1] = cup

    values_per_level = abs(np.prod(data_format))
    if np_array.size % values_per_level > 0:
        if padding:
            values_to_add = ceil(np_array.size / values_per_level) * values_per_level - np_array.size
            np_array = np.append(np_array, np.zeros(values_to_add))
        else:
            return None
    np_array = np_array.reshape(data_format)

    axis_swap = list(range(0, np_array.ndim))
    axis_swap[-2] = axis_swap[-1]
    axis_swap[-1] = axis_swap[-2] - 1

    np_array = np_array.transpose(axis_swap)

    return np_array


def save_h5(file_name, data, labels):
    with h5py.File(file_name, "w") as h5_file:
        h5_file.create_dataset("data", data.shape, dtype='float32')
        h5_file.create_dataset("label", labels.shape, dtype='uint8')
        h5_file["data"][:] = data
        h5_file["label"][:] = labels


def threaded(f, daemon=False):
    import queue
    import threading

    def wrapped_f(q, *args, **kwargs):
        """this function calls the decorated function and puts the
        result in a queue"""
        ret = f(*args, **kwargs)
        q.put(ret)

    def wrap(*args, **kwargs):
        """this is the function returned from the decorator. It fires off
        wrapped_f in a new thread and returns the thread object with
        the result queue attached"""

        q = queue.Queue()

        t = threading.Thread(target=wrapped_f, args=(q,) + args, kwargs=kwargs)
        t.daemon = daemon
        t.start()

        return q

    return wrap

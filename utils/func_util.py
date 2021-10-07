import time
import functools


def time_decorator(func):
    """ A decorator that shows the time consumption of method "func" """
    @functools.wraps(func)
    def inner(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print('Time Consumption: {:.2f}s'.format(end_time - start_time))
        return result
    return inner

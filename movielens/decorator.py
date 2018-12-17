import time


def check_operation_time(func):
    def inner(*args, **kwargs):
        time_start = time.time()
        result = func(*args, **kwargs)
        time_end = time.time()

        print('{} process time is: {} secconds'.format(func.__name__, time_end - time_start))

        return result

    return inner




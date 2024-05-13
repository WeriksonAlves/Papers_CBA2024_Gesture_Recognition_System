from .interfaces import InterfaceTime

import time
import functools

class TimeFunctions(InterfaceTime):
    def tic(self) -> float:
        """Return the current time."""
        return time.perf_counter()
    
    def toc(self, start_time: float) -> float:
        """Return the elapsed time since a given starting time."""
        return time.perf_counter() - start_time
    
    def run_timer(func):
        """Print the runtime of the decorated function."""
        @functools.wraps(func)
        def wrapper_timer(*args, **kwargs):
            start_time = time.perf_counter()
            value = func(*args, **kwargs)
            end_time = time.perf_counter()
            run_time = end_time - start_time
            print(f"Finished {func.__name__}() in {run_time:.4f} secs")
            return value
        return wrapper_timer

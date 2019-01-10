class Timer(object):
    """
    This class can be used to time execution of code surrounded by its
    context.
    """

    def __init__(self):
        self.start_time = 0
        self.end_time = 0
        self.elapsed_time = 0

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        """
        Start timer.
        """
        from time import time
        self.start_time = time()
        return self.start_time

    def stop(self):
        """
        Stop timer.
        """
        from time import time
        self.end_time = time()
        self.elapsed_time = self.end_time - self.start_time
        return self.end_time
class GPUTimer():
    def __init__(self, stream=None):
        self.elapsed_time = 0

        from numba import cuda
        self.start_event = cuda.event(timing=True)
        self.end_event = cuda.event(timing=True)
        self.stream = stream
    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        """
        Start timer.
        """
        if self.stream is None:
            self.start_event.record()
        else:
            self.start_event.record(stream)

    def stop(self):
        """
        Stop timer.
        """
        from numba import cuda

        if self.stream is None:
            self.end_event.record()
            self.end_event.synchronize()
            self.elapsed_time = cuda.event_elapsed_time(self.start_event, self.end_event)
        else:
            self.end_event.record(self.stream)

    def sync():
        if self.stream is None:
            return
        else:
            self.send_event.synchronize()
            self.elapsed_time = cuda.event_elapsed_time(self.start_event, self.end_event)

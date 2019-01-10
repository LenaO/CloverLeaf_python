"""
This module contains functions to log events which can be used for diagnostics
and evaluation of numbers generated during program runtime.
The results may be written to a JSON file.
"""
class EventLog(object):
    """
    Saves events occurring during program runtime
    in form of a hierarchical dictionary.
    """

    def __init__(self, fname=None):
        """
        Create a new instance of the class.

        Args:
            fname (str, optional): If this is given, restore event log from the given JSON file.
        """
        from threading import Lock
        from Timer import Timer
        self.events = {}
        self.lock = Lock()
        self.timer = Timer()
        self.timer.start()

        if fname:
            self.from_file(fname=fname)

    def reset(self):
        """
        Reset events by deleting all entries.
        """


        with self.lock:
            self.events = {}
            self.timer = Timer()
            self.timer.start()

    def add(self, key, value):
        """
        Add a value for a given key.
        If the given key already exists, it will be overwritten.

        Args:
            key (str): Key describing a path to the target dictionary, separated by "/".
            value (object): Value to add, must be serializable.
        """
        with self.lock:
            _add(key, value, self.events)

    def append(self, key, value):
        """
        Append a value for a given key.
        If the given key already exists (and is a list), the value will be appended.

        Args:
            key (str): Key describing a path to the target dictionary, separated by "/".
            value (object): Value to append, must be serializable.
        """
        with self.lock:
            _append(key, value, self.events)

    def get(self, key):
        """
        Get the value for a given key.

        Args:
            key (str): Key describing a path to the target dictionary, separated by "/".

        Returns:
            object: Value found at the given key path.
        """
        return _get(key, self.events)

    def to_str(self):
        """
        Return events as JSON formatted string.

        Returns:
            str: JSON formatted events.
        """
        import json
        from JsonEncoder import JSONEncoder

        self._add_meta()
        return json.dumps(self.events, cls=JSONEncoder)

    def save(self, fname, directory, mode):
        """
        Save content to a file inside the given directory
        with a pre-defined file name.

        Args:
            directory (str): Directory to write file to.
            mode (str): Mode to include in file name and
                        meta data.
        """
        from os.path import join

        self.to_file(join(directory, fname))

    def to_file(self, fname):
        """
        Write content of events to a file in JSON format.

        Args:
            fname (str): Path of file to write content to.
        """
        with open(fname, "w") as f:
            f.write(self.to_str())

        """
        Read content of events from JSON file.

        Args:
            fname (str): Path of file to read content from.
        """
        import json

        with open(fname, "r") as f:
            self.events = json.load(f)

    def _add_meta(self):
        """
        Add meta information.
        """

        self.timer.stop()
        self.add("meta/time", self.timer)


    def __getitem__(self, item):
        """
        Allows indexing this object using [] notation.
        For example,

        event_log["key1"]

        is equivalent to

        event_log.get("key1")

        Args:
            item (str): Key to get content for.

        Returns:
            Item read from dictionary.
        """
        return self.get(key=item)

    def __setitem__(self, key, value):
        """
        Allows indexing this object using [] notation.
        For example,

        event_log["key1"] = 123

        is equivalent to

        event_log.add("key1", 123)

        Args:
            key (str): Key to add content to.
            value (object): Value to add.
        """
        self.add(key=key, value=value)

    def __repr__(self):
        return self.to_str()



# This dictionary contains all events which are logged during program runtime.
events = EventLog()


# -------------------------------------------------------------------
# Functions for easy access to the events hierarchical dictionary
# -------------------------------------------------------------------


def reset():
    """
    Reset events by deleting all entries.
    """
    events.reset()



def add(key, value):
    """
    Add a value for a given key.
    If the given key already exists, it will be overwritten.

    Args:
        key (str): Key describing a path to the target dictionary, separated by "/".
        value (object): Value to add, must be serializable.
    """
    events.add(key, value)


def append(key, value):
    """
    Append a value for a given key.
    If the given key already exists (and is a list), the value will be appended.

    Args:
        key (str): Key describing a path to the target dictionary, separated by "/".
        value (object): Value to append, must be serializable.
    """
    events.append(key, value)


def get(key):
    """
    Get the value for a given key.

    Args:
        key (str): Key describing a path to the target dictionary, separated by "/".

    Returns:
        object: Value found at the given key path.
    """
    return events.get(key)
def to_str():
    """
    Return events as JSON formatted string.

    Returns:
        str: JSON formatted events.
    """
    return events.to_str()


def to_file(fname):
    """
    Write content of events to a file in JSON format.

    Args:
        fname (str): Path of file to write content to.
    """
    events.to_file(fname=fname)


def save(fname, directory, mode):
    """
    Save content to a file inside the given directory
    with a pre-defined file name.

    Args:
        directory (str): Directory to write file to.
        mode (str): Mode to include in file name and
                    meta data.
    """
    events.save(fname,directory=directory,  mode=mode)


# -------------------------------------------------------------------
# Recursive function to work with hierarchical dictionaries
# -------------------------------------------------------------------

def _append(key, value, dictionary):
    """
    Recursive function for append.
    """
    head, tail = _split_key(key)

    if tail:
        if head not in dictionary:
            dictionary[head] = dict()
        _append(tail, value, dictionary[head])
    else:
        if head not in dictionary:
            dictionary[head] = list()
        dictionary[head].append(value)




def _add(key, value, dictionary):
    """
    Recursive function for add.
    """
    head, tail = _split_key(key)

    if tail:
        if head not in dictionary:
            dictionary[head] = dict()
        _add(tail, value, dictionary[head])
    else:
        dictionary[head] = value


def _get(key, dictionary):
    """
    Recursive function for get.
    """
    head, tail = _split_key(key)

    if tail:
        return _get(tail, dictionary[head])
    else:
        return dictionary[head]


def _split_key(key):
    """
    Split a key of different parts separated by "/"
    into a head (part before the first "/") and the
    remaining part.
    """
    split = key.split("/")
    head = split[0]
    tail = split[1:]
    return head, "/".join(tail)


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



import random
import string
import signal
from contextlib import contextmanager
from google.cloud import storage

ID_LEN = 16

def gen_id():
    charset = string.ascii_uppercase + string.ascii_lowercase + string.digits
    chars =  [random.choice(charset) for _ in range(ID_LEN)]
    return ''.join(chars)

@contextmanager
def timeout(time):
    # Register a function to raise a TimeoutError on the signal.
    signal.signal(signal.SIGALRM, raise_timeout)
    # Schedule the signal to be sent after ``time``.
    signal.alarm(time)

    try:
        yield
    finally:
        # Unregister the signal so it won't be triggered
        # if the timeout is not reached.
        signal.signal(signal.SIGALRM, signal.SIG_IGN)

def raise_timeout(signum, frame):
    raise TimeoutError
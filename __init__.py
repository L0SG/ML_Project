import sys
if sys.modules.get("gevent") is not None:
    evented = True
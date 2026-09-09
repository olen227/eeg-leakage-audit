import os as _os
# numba JIT caching fails in the sandbox kernel (no file locator); disable so mne imports.
_os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

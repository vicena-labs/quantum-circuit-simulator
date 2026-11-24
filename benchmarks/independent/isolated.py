"""Run one benchmark worker in its own process group.

A worker that times out is killed with everything it started, and a driver
that is stopped (SIGTERM or Ctrl-C) kills its current worker first, so no
worker outlives the run and competes for CPU with later measurements.
"""
import os, signal, subprocess

_current = None


def _stop(signum, frame):
    if _current is not None and _current.poll() is None:
        os.killpg(_current.pid, signal.SIGKILL)
    raise SystemExit(128 + signum)


def run(cmd, timeout, env=None):
    """(returncode, stdout, stderr); raises subprocess.TimeoutExpired after killing the group."""
    global _current
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    _current = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
                                start_new_session=True)
    try:
        out, err = _current.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(_current.pid, signal.SIGKILL)
        _current.communicate()
        raise
    finally:
        code = _current.returncode
        _current = None
    return code, out, err


def failure(code, err):
    """A worker that printed no result: its exit status and the end of its error output."""
    reason = "killed (likely out of memory)" if code == -signal.SIGKILL else f"exit status {code}"
    return f"{reason}: {err[-300:]}" if err.strip() else reason

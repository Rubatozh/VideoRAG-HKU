"""Runtime seams the multimodal-RAG benchmark installs into this submodule.

Each hook defaults to identity: with nothing installed, behaviour is
unchanged. Call sites add one line after the original expression
(`x = _bench.<hook>(x, ...)`); removing this file and those lines restores
the checkout.
"""

_HOOKS = {}
_FIRED = {}


def install(name, fn):
    """Install a hook. `fn(value, **kw) -> value` must be shape-preserving."""
    _HOOKS[name] = fn


def clear():
    _HOOKS.clear()
    _FIRED.clear()


def installed():
    return sorted(_HOOKS)


def fired():
    """{hook: n_calls}, written onto every record."""
    return dict(_FIRED)


def _apply(name, value, **kw):
    fn = _HOOKS.get(name)
    if fn is None:
        return value                      # <- the upstream path, unchanged
    _FIRED[name] = _FIRED.get(name, 0) + 1
    return fn(value, **kw)


# --------------------------------------------------------------- the seams
def frame_times(segment_times_info, **kw):
    """INPUT POOL. `split_video` picks frames per segment as absolute seconds;
    this lets the harness substitute the pool's own, leaving segment naming and
    per-segment audio extraction untouched."""
    return _apply("frame_times", segment_times_info, **kw)


def evidence(prompt, **kw):
    """OUTPUT. The assembled context immediately before the answerer reads it."""
    return _apply("evidence", prompt, **kw)


def frames_for(frame_times, **kw):
    """PIXELS. Return replacement PIL frames for `frame_times`, or None to
    fall back to the upstream decode."""
    fn = _HOOKS.get("frames_for")
    if fn is None:
        return None                       # <- the upstream path, unchanged
    _FIRED["frames_for"] = _FIRED.get("frames_for", 0) + 1
    return fn(frame_times, **kw)

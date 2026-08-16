"""Seams the multimodal-RAG benchmark installs at runtime.

ADDED BY THE BENCHMARK. Not upstream. See bench/systems/hku.py.

Every hook DEFAULTS TO IDENTITY. With nothing installed this file changes no
behaviour at all, which is the point: the upstream execution path is not
"maintained alongside" the new one, it simply IS the path whenever the harness
has not installed a hook. There is no second branch to keep in sync and nothing
to bit-rot.

THE EDIT CONTRACT, so this stays revertible:

  * every call site is ONE inserted line, applied AFTER the original expression,
    of the form  `x = _bench.<hook>(x, ...)`
  * the original expression is never modified, moved, or wrapped in a condition
  * removing this file and stripping lines containing `_bench.` restores the
    checkout byte-for-byte

`bench/tests/submodule_seams.py` asserts exactly that, so the fallback claim is
checked rather than promised.

WHY HOOKS FIRE IS RECORDED. `fired()` counts each hook's invocations and the
harness writes it onto every record. A hook that silently failed to install
looks identical to one that installed and did nothing -- that is how `vgent`
came to accept `--pool-rate`, name its arm for it, and ignore it.
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
    """{hook: n_calls}. Written onto the record so an arm proves its path."""
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


def segments(retrieved, **kw):
    """SELECT. The retrieved segment ids, after the entity/visual union."""
    return _apply("segments", retrieved, **kw)


def evidence(prompt, **kw):
    """OUTPUT. The assembled context immediately before the answerer reads it."""
    return _apply("evidence", prompt, **kw)


def frames_for(frame_times, **kw):
    """PIXELS. Return replacement PIL frames for `frame_times`, or None.

    Unlike the other hooks this one may return None, meaning "not handled --
    run upstream". That is what lets ONE seam cover both captioners: an
    unknown video falls through instead of being silently substituted.

    It exists because `frame_times` alone was not enough. `encode_video` is
    the single place both `segment_caption` (index time) and
    `retrieved_segment_caption` (QUERY time) decode pixels, and the query-time
    one computed its own `np.linspace` over the raw video -- so hku's answer
    evidence came from moments the benchmark's input pool never authorised.
    Substituting here unifies the moments AND the pixels, and costs less than
    upstream: a cached JPEG read instead of moviepy random access.
    """
    fn = _HOOKS.get("frames_for")
    if fn is None:
        return None                       # <- the upstream path, unchanged
    _FIRED["frames_for"] = _FIRED.get("frames_for", 0) + 1
    return fn(frame_times, **kw)

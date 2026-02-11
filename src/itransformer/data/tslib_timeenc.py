from __future__ import annotations


_VALID_POLICIES = {"strict", "auto", "off"}


def resolve_tslib_timeenc(cfg, requested_timeenc, *, source: str) -> int:
    """Resolve timeenc for TSLIB variant according to embed/timeenc policy."""
    try:
        requested = int(requested_timeenc)
    except Exception as exc:
        raise ValueError(
            f"[data] Invalid timeenc value at {source}: {requested_timeenc!r}"
        ) from exc

    variant = str(getattr(getattr(cfg, "model", None), "variant", "") or "")
    if variant != "TSLIB":
        return requested

    tcfg = getattr(getattr(cfg, "model", None), "tslib", None)
    embed = str(getattr(tcfg, "embed", "timeF") or "timeF")
    policy = str(getattr(tcfg, "timeenc_policy", "strict") or "strict").lower().strip()
    if policy not in _VALID_POLICIES:
        allowed = ", ".join(sorted(_VALID_POLICIES))
        raise ValueError(
            f"[data] Unsupported model.tslib.timeenc_policy={policy!r}. "
            f"Allowed values: {allowed}"
        )

    expected = 1 if embed == "timeF" else 0
    if policy == "off" or requested == expected:
        return requested

    msg = (
        f"[data] TSLIB time encoding mismatch at {source}: "
        f"model.tslib.embed={embed!r} expects timeenc={expected}, got {requested}."
    )
    if policy == "strict":
        raise ValueError(
            msg + f" Fix by setting data.timeenc={expected} "
            "or model.tslib.timeenc_policy=auto/off."
        )

    print(msg + f" Auto-correcting to {expected} (model.tslib.timeenc_policy=auto).")
    return expected

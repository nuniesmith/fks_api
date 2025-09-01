def test_config_stubs_or_shared():
    """Service should function whether or not shared_python is installed.

    Preference order:
    1. shared_python when USE_SHARED=1 and import succeeds
    2. fallback standalone_shared stubs
    """
    import os
    use_shared = os.getenv("USE_SHARED", "0") == "1"

    if use_shared:
        try:
            from shared_python import load_config  # type: ignore
        except Exception:
            # Even if requested, absence should not crash test— fallback
            from standalone_shared import load_config  # type: ignore
    else:
        from standalone_shared import load_config  # type: ignore

    cfg = load_config()
    assert isinstance(cfg, dict)
    assert "environment" in cfg

def test_shared_import():
    try:
        from shared_python import load_config, get_settings, get_risk_threshold  # type: ignore
    except Exception as e:  # pragma: no cover
        raise AssertionError(f"Failed to import shared package: {e}")
    assert callable(load_config)

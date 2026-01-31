from pm_rul.cli import _health_status


def test_health_status_string():
    status = _health_status()
    assert isinstance(status, str)

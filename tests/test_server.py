"""Server endpoint regression tests.

Catches the failure mode where server/main.py imports cleanly but an
endpoint crashes at runtime (e.g. NameError from missing import).
Each endpoint is exercised with a real TestClient request and response
structure is validated — not just status code.
"""
import sys
sys.path.insert(0, ".")

import pytest
from fastapi.testclient import TestClient

from server.main import app


@pytest.fixture
def client():
    return TestClient(app)


class TestHealth:
    def test_health_returns_ok(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


class TestSignal:
    def test_signal_returns_valid_structure(self, client):
        r = client.get("/api/signal")
        assert r.status_code == 200
        d = r.json()
        assert "date" in d
        assert "signals" in d
        assert "action" in d
        assert isinstance(d["signals"], list)

    def test_signal_reason_is_3factor_format(self, client):
        """No stale 5-factor 'macd+kdj' references in reason strings."""
        r = client.get("/api/signal")
        for s in r.json().get("signals", []):
            assert "macd+kdj" not in s["reason"], (
                f"Stale reason format in {s['symbol']}: {s['reason']}"
            )

    def test_signal_fields_present(self, client):
        r = client.get("/api/signal")
        for s in r.json().get("signals", []):
            for field in ("symbol", "score", "price", "shares", "reason"):
                assert field in s, f"Missing field '{field}' in signal for {s.get('symbol', '?')}"


class TestDecision:
    def test_decision_returns_valid_structure(self, client):
        r = client.get("/api/decision")
        assert r.status_code == 200
        d = r.json()
        assert "selected" in d
        assert "rankings" in d
        assert "date" in d

    def test_decision_rankings_have_required_fields(self, client):
        r = client.get("/api/decision")
        for rk in r.json().get("rankings", []):
            for field in ("symbol", "score", "ma120", "above_ma", "rising_ma"):
                assert field in rk, f"Missing field '{field}' in ranking for {rk.get('symbol', '?')}"


class TestHistory:
    def test_history_returns_entries(self, client):
        r = client.get("/api/history", params={"days": 30})
        assert r.status_code == 200
        d = r.json()
        assert "history" in d
        assert len(d["history"]) > 0

    def test_history_top3_structure(self, client):
        r = client.get("/api/history", params={"days": 30})
        for h in r.json().get("history", []):
            assert "date" in h
            assert "top3" in h
            for t in h["top3"]:
                assert "symbol" in t
                assert "score" in t


class TestKlines:
    def test_klines_returns_all_indicator_fields(self, client):
        r = client.get("/api/klines/510300", params={"days": 30})
        assert r.status_code == 200
        d = r.json()
        for field in ("dates", "klines",
                      "ma5", "ma10", "ma20", "ma120", "ma250",
                      "macd_dif", "macd_dea", "macd_hist",
                      "kdj_k", "kdj_d", "kdj_j"):
            assert field in d, f"Missing field '{field}' in klines response"
            assert isinstance(d[field], list), f"'{field}' is not a list"

    def test_klines_macd_has_valid_data(self, client):
        r = client.get("/api/klines/510300", params={"days": 60})
        d = r.json()
        valid = [v for v in d["macd_dif"] if v is not None]
        assert len(valid) > 0, "MACD DIF has no valid values"

    def test_klines_kdj_has_valid_data(self, client):
        r = client.get("/api/klines/510300", params={"days": 60})
        d = r.json()
        valid = [v for v in d["kdj_k"] if v is not None]
        assert len(valid) > 0, "KDJ K has no valid values"

    def test_klines_weekly_period(self, client):
        r = client.get("/api/klines/510300", params={"days": 30, "period": "weekly"})
        assert r.status_code == 200
        assert r.json()["period"] == "weekly"

    def test_klines_monthly_period(self, client):
        r = client.get("/api/klines/510300", params={"days": 120, "period": "monthly"})
        assert r.status_code == 200
        assert r.json()["period"] == "monthly"

    def test_klines_unknown_symbol_returns_empty(self, client):
        r = client.get("/api/klines/UNKNOWN", params={"days": 30})
        assert r.status_code == 200
        assert r.json()["klines"] == []


class TestCache:
    def test_cache_clear_succeeds(self, client):
        r = client.post("/api/cache/clear")
        assert r.status_code == 200
        assert r.json()["status"] == "cleared"

    def test_signal_still_works_after_cache_clear(self, client):
        client.post("/api/cache/clear")
        r = client.get("/api/signal")
        assert r.status_code == 200
        assert len(r.json()["signals"]) >= 0

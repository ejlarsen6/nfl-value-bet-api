import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
 
import main
 
 
class FakeModel:
    def predict_proba(self, X):
        return np.array([[0.35, 0.65]])  # always predicts 65% home win
 
 
@pytest.fixture
def client(monkeypatch):
    fake_features = pd.DataFrame([
        {"season": 2026, "week": 1, "home_team": "KC", "away_team": "BUF",
         "home_devigged_prob": 0.58,
         "home_passing_epa_per_play_roll3": 0.1, "home_passing_epa_per_play_roll8": 0.1,
         "home_rushing_epa_per_play_roll3": 0.1, "home_rushing_epa_per_play_roll8": 0.1,
         "away_passing_epa_per_play_roll3": 0.1, "away_passing_epa_per_play_roll8": 0.1,
         "away_rushing_epa_per_play_roll3": 0.1, "away_rushing_epa_per_play_roll8": 0.1},
    ]).set_index(["season", "week", "home_team", "away_team"])
 
    # Substitute the real model/features with known, controlled fakes
    monkeypatch.setattr(main, "model", FakeModel())
    monkeypatch.setattr(main, "features_df", fake_features)
    monkeypatch.setattr(main, "MODEL_LOADED", True)
    monkeypatch.setattr(main, "FEATURES_LOADED", True)
 
    return TestClient(main.app)
 
 
def test_health_reports_ok_when_loaded(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["games_available"] == 1
 
 
def test_health_reports_503_when_model_not_loaded(client, monkeypatch):
    monkeypatch.setattr(main, "MODEL_LOADED", False)
    monkeypatch.setattr(main, "MODEL_LOAD_ERROR", "fake load failure", raising=False)
    resp = client.get("/health")
    assert resp.status_code == 503
 
 
def test_predict_known_matchup_returns_expected_shape(client):
    resp = client.post("/predict", json={
        "season": 2026, "week": 1, "home_team": "KC", "away_team": "BUF"
    })
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"model_home_win_prob", "market_home_win_prob", "edge", "value_flag"}
 
 
def test_predict_computes_edge_correctly(client):
    # FakeModel always returns 0.65; fixture's market prob is 0.58
    resp = client.post("/predict", json={
        "season": 2026, "week": 1, "home_team": "KC", "away_team": "BUF"
    })
    body = resp.json()
    assert body["model_home_win_prob"] == 0.65
    assert body["market_home_win_prob"] == 0.58
    assert abs(body["edge"] - 0.07) < 1e-6
    assert body["value_flag"] is True  # 0.07 edge >= 0.05 threshold
 
 
def test_predict_unknown_matchup_returns_404(client):
    resp = client.post("/predict", json={
        "season": 2026, "week": 99, "home_team": "KC", "away_team": "BUF"
    })
    assert resp.status_code == 404
    assert "No features found" in resp.json()["detail"]
 
 
def test_predict_missing_field_returns_422(client):
    resp = client.post("/predict", json={
        "season": 2026, "week": 1, "home_team": "KC" # Missing away_team
    })
    assert resp.status_code == 422
 
 
def test_predict_wrong_type_returns_422(client):
    # season should be an int, not a string
    resp = client.post("/predict", json={
        "season": "not-a-year", "week": 1, "home_team": "KC", "away_team": "BUF"
    })
    assert resp.status_code == 422
 
 
@pytest.mark.parametrize("edge_case_prob,expected_flag", [
    (0.60, True),   # 0.65 - 0.60 = 0.05 
    (0.61, False),  # 0.65 - 0.61 = 0.04 
])
def test_value_flag_threshold_boundary(client, monkeypatch, edge_case_prob, expected_flag):
    fake_features = pd.DataFrame([
        {"season": 2026, "week": 1, "home_team": "KC", "away_team": "BUF",
         "home_devigged_prob": edge_case_prob,
         "home_passing_epa_per_play_roll3": 0.1, "home_passing_epa_per_play_roll8": 0.1,
         "home_rushing_epa_per_play_roll3": 0.1, "home_rushing_epa_per_play_roll8": 0.1,
         "away_passing_epa_per_play_roll3": 0.1, "away_passing_epa_per_play_roll8": 0.1,
         "away_rushing_epa_per_play_roll3": 0.1, "away_rushing_epa_per_play_roll8": 0.1},
    ]).set_index(["season", "week", "home_team", "away_team"])
    monkeypatch.setattr(main, "features_df", fake_features)
 
    resp = client.post("/predict", json={
        "season": 2026, "week": 1, "home_team": "KC", "away_team": "BUF"
    })
    assert resp.json()["value_flag"] == expected_flag
 
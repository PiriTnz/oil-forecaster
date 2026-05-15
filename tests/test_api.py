"""Tests for FastAPI endpoints"""
import pytest
from fastapi.testclient import TestClient
from src.api.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "oil-forecaster" in r.json()["service"]


def test_health(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_list_events(client):
    r = client.get("/api/v1/data/events")
    assert r.status_code == 200
    data = r.json()
    assert data["count"] > 0
    assert isinstance(data["events"], list)


def test_list_events_filtered_by_type(client):
    r = client.get("/api/v1/data/events?event_type=war")
    assert r.status_code == 200
    for e in r.json()["events"]:
        assert e["event_type"] == "war"


def test_list_events_filtered_by_year(client):
    r = client.get("/api/v1/data/events?start_year=2008&end_year=2009")
    assert r.status_code == 200
    for e in r.json()["events"]:
        year = int(e["start_date"][:4])
        assert 2008 <= year <= 2009


def test_active_events(client):
    r = client.get("/api/v1/data/events/active")
    assert r.status_code == 200
    assert "events" in r.json()


def test_metrics_exposed(client):
    r = client.get("/metrics")
    assert r.status_code == 200

"""
tests/test_research_endpoints.py - Integration tests for the new NLP Research Hub API endpoints.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from fastapi.testclient import TestClient
from api.main import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

class TestResearchEndpoints:
    def test_get_leaderboard(self, client):
        response = client.get("/research/leaderboard")
        assert response.status_code == 200
        data = response.json()
        assert "metadata" in data
        assert "leaderboard" in data
        assert isinstance(data["leaderboard"], list)
        if len(data["leaderboard"]) > 0:
            item = data["leaderboard"][0]
            assert "key" in item
            assert "name" in item
            assert "group" in item
            assert "rougeL" in item
            assert "bertscore" in item
            assert "latency" in item
            assert "faithfulness" in item
            assert "hallucination_pct" in item

    def test_get_benchmark_samples(self, client):
        response = client.get("/research/benchmark/samples?page=1&limit=5")
        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "page" in data
        assert "limit" in data
        assert "pages" in data
        assert "items" in data
        assert isinstance(data["items"], list)
        assert len(data["items"]) <= 5
        if len(data["items"]) > 0:
            sample = data["items"][0]
            assert "id" in sample
            assert "title" in sample
            assert "category" in sample
            assert "article" in sample
            assert "summary" in sample
            assert "models" in sample

    def test_get_benchmark_samples_filter(self, client):
        response = client.get("/research/benchmark/samples?category=Short&limit=2")
        assert response.status_code == 200
        data = response.json()
        for item in data["items"]:
            assert item["category"] == "Short"

    def test_get_hybrid_study(self, client):
        response = client.get("/research/hybrid-study")
        assert response.status_code == 200
        data = response.json()
        assert "groups" in data
        assert "extractive" in data["groups"]
        assert "abstractive" in data["groups"]
        assert "hybrid" in data["groups"]
        assert "long_document_analysis" in data
        assert "insights" in data["long_document_analysis"]

    def test_get_report(self, client):
        response = client.get("/research/report")
        assert response.status_code == 200
        data = response.json()
        assert "title" in data
        assert "author" in data
        assert "conclusions" in data
        assert isinstance(data["conclusions"], list)
        assert len(data["conclusions"]) > 0
        assert "metrics_summary" in data

    def test_run_benchmark(self, client):
        response = client.post("/research/benchmark/run")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "message" in data

    def test_get_leaderboard_by_category(self, client):
        response = client.get("/research/leaderboard/by-category?category=Short")
        assert response.status_code == 200
        data = response.json()
        assert data["category"] == "Short"
        assert "total_samples" in data
        assert "leaderboard" in data
        assert isinstance(data["leaderboard"], list)
        if len(data["leaderboard"]) > 0:
            item = data["leaderboard"][0]
            assert "key" in item
            assert "composite" in item


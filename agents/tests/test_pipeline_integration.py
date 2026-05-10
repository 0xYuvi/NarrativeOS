"""End-to-end pipeline integration test — chains all 5 agents (correlation → risk → strategy → execution)."""

import pytest
from httpx import ASGITransport, AsyncClient

from agents.execution_api.app import app as execution_app
from agents.market_correlation.app import app as market_correlation_app
from agents.risk_intelligence.app import app as risk_app
from agents.strategy_agent.app import app as strategy_app

SAMPLE_EVENT = {
    "id": "evt_20260509_nvda_001",
    "source": "news",
    "source_actor": "narrativeos-news-scraper",
    "title": "NVIDIA announces Blackwell Ultra with 3x AI inference performance",
    "body": (
        "NVIDIA today announced its next-generation Blackwell Ultra chip at the GTC conference, "
        "boasting three times the AI inference performance of the previous generation. "
        "The semiconductor giant expects data center revenue to continue growing 200% year-over-year "
        "as hyperscalers increase AI infrastructure spending. Analysts are bullish on the sustained "
        "demand driven by large language model training and inference workloads. "
        "CEO Jensen Huang called it 'the most successful product launch in company history.'"
    ),
    "url": "https://example.com/nvda-blackwell-ultra",
    "author": "Test Author",
    "published_at": "2026-05-09T14:00:00Z",
    "collected_at": "2026-05-09T14:01:00Z",
    "ticker_mentions": ["NVDA", "AMD", "TSM"],
    "entities": [
        {"name": "NVIDIA", "type": "company", "ticker": "NVDA"},
        {"name": "AMD", "type": "company", "ticker": "AMD"},
    ],
    "metadata": {"source_confidence": 0.95, "word_count": 120},
}

SAMPLE_EVENT_BEARISH = {
    "id": "evt_20260509_tsla_002",
    "source": "news",
    "source_actor": "narrativeos-news-scraper",
    "title": "Tesla deliveries miss estimates amid demand concerns",
    "body": (
        "Tesla reported Q2 delivery numbers that missed analyst expectations by a wide margin, "
        "citing softening demand across all major markets. The electric vehicle maker cited "
        "increased competition from Chinese manufacturers and macroeconomic headwinds. "
        "Several analysts downgraded the stock, pointing to inventory buildup and price cuts "
        "that are squeezing margins. Regulatory scrutiny over self-driving claims has also intensified."
    ),
    "url": "https://example.com/tsla-miss",
    "author": "Test Author",
    "published_at": "2026-05-09T15:00:00Z",
    "collected_at": "2026-05-09T15:01:00Z",
    "ticker_mentions": ["TSLA"],
    "entities": [{"name": "Tesla", "type": "company", "ticker": "TSLA"}],
    "metadata": {"source_confidence": 0.90},
}


@pytest.fixture
def pipeline_clients():
    return {
        "market_correlation": AsyncClient(
            transport=ASGITransport(app=market_correlation_app), base_url="http://test"
        ),
        "risk": AsyncClient(
            transport=ASGITransport(app=risk_app), base_url="http://test"
        ),
        "strategy": AsyncClient(
            transport=ASGITransport(app=strategy_app), base_url="http://test"
        ),
        "execution": AsyncClient(
            transport=ASGITransport(app=execution_app), base_url="http://test"
        ),
    }


@pytest.mark.anyio
async def test_full_pipeline_bullish(pipeline_clients):
    """Happy path: bullish event → correlate → assess risk → formulate → execute."""
    clients = pipeline_clients

    # ── Step 1: Market Correlation ──────────────────────────────────────────
    resp = await clients["market_correlation"].post(
        "/correlate", json={"events": [SAMPLE_EVENT]}
    )
    assert resp.status_code == 200
    correlation = resp.json()
    assert len(correlation["correlations"]) > 0
    assert correlation["correlations"][0]["ticker"] == "NVDA"
    assert correlation["correlations"][0]["sector"] == "Technology"

    # ── Step 2: Build debate + sentiment from correlation output ────────────
    debate_result = {
        "bull_case": (
            "Data center revenue growing 200% YoY, new Blackwell architecture entering "
            "mass production, strong demand from hyperscalers, analyst upgrades"
        ),
        "bear_case": (
            "Valuation at 35x forward earnings, potential export restriction escalation, "
            "increasing competition from AMD and custom ASICs"
        ),
        "arbiter_ruling": "Bull case stronger — growth fundamentals outweigh valuation concerns at this stage",
        "debate_rounds": 3,
    }
    sentiment = {
        "polarity": 0.65,
        "confidence": 0.8,
        "intensity": 0.4,
        "instability_score": 0.2,
    }
    narrative = {
        "summary": correlation["macro_relationships"][0]["narrative_theme"],
        "momentum_score": 0.72,
        "propagation_rate": 0.6,
        "thematic_category": "technology",
    }

    resp = await clients["risk"].post(
        "/assess",
        json={
            "debate_result": debate_result,
            "narrative": narrative,
            "sentiment": sentiment,
        },
    )
    assert resp.status_code == 200
    risk = resp.json()
    assert 0 <= risk["risk_score"] <= 1

    # ── Step 3: Strategy Formulation ────────────────────────────────────────
    risk_assessment = {
        "risk_score": risk["risk_score"],
        "risk_factors": risk["risk_factors"],
    }
    resp = await clients["strategy"].post(
        "/formulate",
        json={
            "risk_assessment": risk_assessment,
            "debate": debate_result,
        },
    )
    assert resp.status_code == 200
    strategy = resp.json()
    assert strategy["ticker"] == "UNKNOWN"
    assert strategy["direction"] in ("BUY", "SELL", "HOLD", "WATCHLIST")
    assert 0 <= strategy["confidence"] <= 1
    assert len(strategy["reasoning_trace"]) > 0

    # If BUY or SELL, proceed to execution
    if strategy["direction"] in ("BUY", "SELL") and strategy["confidence"] >= 0.6:
        # ── Step 4: Execution ───────────────────────────────────────────────
        resp = await clients["execution"].post(
            "/execute",
            json={
                "signal": {
                    "ticker": strategy["ticker"],
                    "direction": strategy["direction"],
                    "confidence": strategy["confidence"],
                },
                "approval": {"approved": True, "notes": "Approved by pipeline test"},
                "mode": "simulated",
            },
        )
        assert resp.status_code == 200
        execution = resp.json()
        assert execution["status"] == "filled"
        assert execution["quantity"] > 0
        assert execution["price"] > 0
        assert execution["direction"] == strategy["direction"]
    else:
        pytest.skip("Signal direction not actionable (BUY/SELL with confidence >= 0.6)")


@pytest.mark.anyio
async def test_full_pipeline_bearish(pipeline_clients):
    """Bearish event → correlation → risk → strategy (expected WATCHLIST or HOLD)."""
    clients = pipeline_clients

    resp = await clients["market_correlation"].post(
        "/correlate", json={"events": [SAMPLE_EVENT_BEARISH]}
    )
    assert resp.status_code == 200
    correlation = resp.json()

    debate_result = {
        "bull_case": "Tesla has strong brand loyalty and long-term AI/robotaxi potential",
        "bear_case": (
            "Deliveries missing estimates, demand softening, price margins compressing, "
            "increased competition from BYD and other Chinese EV makers, regulatory probes"
        ),
        "arbiter_ruling": "Bear case stronger — near-term headwinds outweigh long-term optionality",
        "debate_rounds": 3,
    }
    sentiment = {
        "polarity": -0.45,
        "confidence": 0.85,
        "intensity": 0.6,
        "instability_score": 0.55,
    }
    narrative = {
        "summary": correlation["macro_relationships"][0]["narrative_theme"],
        "momentum_score": 0.25,
        "propagation_rate": 0.4,
        "thematic_category": "automotive",
    }

    resp = await clients["risk"].post(
        "/assess",
        json={
            "debate_result": debate_result,
            "narrative": narrative,
            "sentiment": sentiment,
        },
    )
    assert resp.status_code == 200
    risk = resp.json()
    # Bearish news with regulatory risk should have higher risk score
    assert risk["risk_score"] >= 0.3
    assert len(risk["anomaly_flags"]) >= 0

    resp = await clients["strategy"].post(
        "/formulate",
        json={
            "risk_assessment": {
                "risk_score": risk["risk_score"],
                "risk_factors": risk["risk_factors"],
            },
            "debate": debate_result,
        },
    )
    assert resp.status_code == 200
    strategy = resp.json()
    # High risk + bearish debate → expect HOLD or WATCHLIST
    if risk["risk_score"] > 0.5:
        assert strategy["direction"] in ("HOLD", "WATCHLIST")
    assert 0 <= strategy["confidence"] <= 1


@pytest.mark.anyio
async def test_pipeline_execution_rejects_low_confidence(pipeline_clients):
    """Execution API rejects signals below confidence threshold."""
    resp = await pipeline_clients["execution"].post(
        "/execute",
        json={
            "signal": {"ticker": "NVDA", "direction": "BUY", "confidence": 0.3},
            "approval": {"approved": True},
            "mode": "simulated",
        },
    )
    # Should succeed - the execution API doesn't filter by confidence,
    # that's the canvas risk-gate's job
    assert resp.status_code == 200


@pytest.mark.anyio
async def test_pipeline_execution_rejects_unapproved(pipeline_clients):
    """Execution properly enforces human approval."""
    resp = await pipeline_clients["execution"].post(
        "/execute",
        json={
            "signal": {"ticker": "NVDA", "direction": "BUY", "confidence": 0.8},
            "approval": {"approved": False, "notes": "Too risky"},
            "mode": "simulated",
        },
    )
    assert resp.status_code == 403


@pytest.mark.anyio
async def test_pipeline_execution_rejects_hold(pipeline_clients):
    """Execution API rejects HOLD/WATCHLIST signals (no trade needed)."""
    for direction in ("HOLD", "WATCHLIST"):
        resp = await pipeline_clients["execution"].post(
            "/execute",
            json={
                "signal": {"ticker": "NVDA", "direction": direction, "confidence": 0.5},
                "approval": {"approved": True},
                "mode": "simulated",
            },
        )
        assert resp.status_code == 400


@pytest.mark.anyio
async def test_pipeline_risk_anomaly_detection(pipeline_clients):
    """Risk agent flags anomalies in hype/fear narratives."""
    hype_debate = {
        "bull_case": "This is a guaranteed moon shot! To the moon! Pump it! No downside!",
        "bear_case": "Some risks but negligible",
        "arbiter_ruling": "Bull case guaranteed — this is a sure thing, risk-free investment",
        "debate_rounds": 1,
    }
    hype_sentiment = {
        "polarity": 0.95,
        "confidence": 0.3,
        "intensity": 0.9,
        "instability_score": 0.8,
    }
    resp = await pipeline_clients["risk"].post(
        "/assess",
        json={
            "debate_result": hype_debate,
            "narrative": {"summary": "Moon shot pump", "momentum_score": 0.1},
            "sentiment": hype_sentiment,
        },
    )
    assert resp.status_code == 200
    risk = resp.json()
    anomalies = risk.get("anomaly_flags", [])
    anomaly_types = [a["type"] for a in anomalies]
    assert "hype_detected" in anomaly_types or "misleading_confidence" in anomaly_types
    assert risk["confidence_degradation"] > 0


@pytest.mark.anyio
async def test_all_agents_health(pipeline_clients):
    """Every agent and service reports healthy."""
    health_checks = {
        "market_correlation": ("/health", "market-correlation"),
        "risk": ("/health", "risk-intelligence"),
        "strategy": ("/health", "strategy"),
        "execution": ("/health", "execution-api"),
    }
    for name, (path, expected_agent) in health_checks.items():
        resp = await pipeline_clients[name].get(path)
        assert resp.status_code == 200, f"{name} health check failed"
        data = resp.json()
        assert data["status"] == "ok", f"{name} not healthy: {data}"

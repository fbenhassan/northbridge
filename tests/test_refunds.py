from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
from fastapi.testclient import TestClient

from backend import db


@pytest.fixture
def refund_client(monkeypatch, tmp_path):
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "refunds.sqlite3")

    from backend.main import app

    with TestClient(app) as client:
        conn = db.get_connection()
        try:
            conn.execute(
                "INSERT INTO orders (order_id, status, total, notes) "
                "VALUES (?, ?, ?, ?)",
                ("TEST-REFUND", "shipped", 100.00, ""),
            )
            conn.commit()
        finally:
            conn.close()
        yield client


@pytest.mark.parametrize("amount", [0, -0.01])
def test_rejects_nonpositive_refund_amounts(refund_client, amount):
    response = refund_client.post(
        "/orders/TEST-REFUND/refund", json={"amount": amount}
    )

    assert response.status_code == 422
    assert db.total_refunded("TEST-REFUND") == 0


def test_accepts_exact_remaining_balance(refund_client):
    partial = refund_client.post(
        "/orders/TEST-REFUND/refund", json={"amount": 25.25}
    )
    remaining = refund_client.post(
        "/orders/TEST-REFUND/refund", json={"amount": 74.75}
    )

    assert partial.status_code == 200
    assert remaining.status_code == 200
    assert remaining.json()["total_refunded"] == 100.0
    assert db.total_refunded("TEST-REFUND") == 100.0


def test_rejects_one_cent_over_remaining_balance(refund_client):
    first = refund_client.post(
        "/orders/TEST-REFUND/refund", json={"amount": 99.99}
    )
    over = refund_client.post(
        "/orders/TEST-REFUND/refund", json={"amount": 0.02}
    )

    assert first.status_code == 200
    assert over.status_code == 400
    assert db.total_refunded("TEST-REFUND") == 99.99


def test_accepts_cumulative_partial_refunds_up_to_total(refund_client):
    amounts = [12.34, 56.78, 30.88]
    responses = [
        refund_client.post(
            "/orders/TEST-REFUND/refund", json={"amount": amount}
        )
        for amount in amounts
    ]

    assert [response.status_code for response in responses] == [200, 200, 200]
    assert responses[-1].json()["total_refunded"] == 100.0
    assert refund_client.post(
        "/orders/TEST-REFUND/refund", json={"amount": 0.01}
    ).status_code == 400


def test_concurrent_refunds_cannot_exceed_order_total(refund_client):
    barrier = Barrier(2)

    def submit_refund():
        barrier.wait()
        return refund_client.post(
            "/orders/TEST-REFUND/refund", json={"amount": 60.00}
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        responses = list(executor.map(lambda _: submit_refund(), range(2)))

    assert sorted(response.status_code for response in responses) == [200, 400]
    assert db.total_refunded("TEST-REFUND") == 60.0

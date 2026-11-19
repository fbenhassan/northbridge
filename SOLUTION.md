# Refund integrity

- **Reproduced defect:** On a $100 order, the prior endpoint accepted a $75 refund followed by $50, recording $125 total. It inserted each request without checking the order balance; `amount` was an unconstrained float.
- **Invariant:** Cumulative successful refunds for an order must be less than or equal to its total.
- **Partial refunds:** Each refund is checked against the remaining balance, not against the original total. Any number of positive partial refunds remains valid until they sum to the order total.
- **Implementation:** Validate finite positive amounts at the API boundary, then in `record_refund` load the order, sum existing refunds, compare the proposed amount with the exact remaining balance using `Decimal`, and insert only if it fits. Missing orders return 404; invalid or excess refunds return 400 (invalid request values are rejected with 422).
- **Concurrency:** `BEGIN IMMEDIATE` acquires SQLite's write reservation before reading the order and refund total, serializing competing refund transactions so they cannot both approve against a stale balance.
- **Files changed:** `backend/db.py`, `backend/main.py`, `tests/test_refunds.py`, `SOLUTION.md`.
- **Tests executed:** `/workspaces/northbridge/.venv/bin/pytest -q tests/test_refunds.py` — **6 passed**. One upstream Starlette deprecation warning.
- **Manual boundary test:** For a $100 order, submit $25.25, then $74.75 (both should succeed; cumulative total $100), then $0.01 (should return 400 and leave the total unchanged). The automated suite exercised the exact-balance and over-balance boundaries; this sequence was not run separately by hand.
- **Known limitations:** Amounts are still stored as SQLite `REAL` and API totals are returned as floats, so this is not end-to-end fixed-point currency storage. There is no currency/decimal-scale validation or idempotency key; requests from code that bypasses this transaction can also violate the invariant.

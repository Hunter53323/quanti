"""Tests for order state machine."""

import tempfile
from pathlib import Path

import pytest

from quanti.execution.order_manager import TRANSITIONS, OrderManager, OrderStateError, OrderStatus
from quanti.state.journal import Journal


class TestOrderStateTransitions:
    """Verify the state machine allows correct transitions."""

    def test_new_to_submitted(self):
        assert OrderStatus.SUBMITTED in TRANSITIONS[OrderStatus.NEW]

    def test_submitted_to_acknowledged(self):
        assert OrderStatus.ACKNOWLEDGED in TRANSITIONS[OrderStatus.SUBMITTED]

    def test_submitted_to_rejected(self):
        assert OrderStatus.REJECTED in TRANSITIONS[OrderStatus.SUBMITTED]

    def test_acknowledged_to_filled(self):
        assert OrderStatus.FILLED in TRANSITIONS[OrderStatus.ACKNOWLEDGED]

    def test_acknowledged_to_cancelled(self):
        assert OrderStatus.CANCELLED in TRANSITIONS[OrderStatus.ACKNOWLEDGED]

    def test_partial_to_filled(self):
        assert OrderStatus.FILLED in TRANSITIONS[OrderStatus.PARTIAL_FILLED]

    def test_terminal_no_transitions(self):
        """Terminal states have no outgoing transitions."""
        assert len(TRANSITIONS[OrderStatus.FILLED]) == 0
        assert len(TRANSITIONS[OrderStatus.CANCELLED]) == 0
        assert len(TRANSITIONS[OrderStatus.REJECTED]) == 0


class TestOrderManager:
    """Integration tests for OrderManager with real Journal."""

    def test_create_and_submit(self, journal):
        om = OrderManager(journal)
        om.create_order("test-001", "510300.SH", "buy", 1000, 3.50, "limit")

        order = journal.get_order("test-001")
        assert order is not None
        assert order["status"] == OrderStatus.NEW.value
        assert order["symbol"] == "510300.SH"
        assert order["quantity"] == 1000

        # Submit
        om.submit("test-001")
        order = journal.get_order("test-001")
        assert order["status"] == OrderStatus.SUBMITTED.value

    def test_full_lifecycle(self, journal):
        om = OrderManager(journal)
        om.create_order("test-002", "159915.SH", "sell", 500, 2.80, "limit")

        om.submit("test-002")
        om.acknowledge("test-002")
        om.fill("test-002", filled_qty=500, avg_fill_price=2.81, is_complete=True)

        order = journal.get_order("test-002")
        assert order["status"] == OrderStatus.FILLED.value
        assert order["filled_qty"] == 500
        assert order["avg_fill_price"] == 2.81

    def test_partial_fill_then_complete(self, journal):
        om = OrderManager(journal)
        om.create_order("test-003", "510500.SH", "buy", 2000, 6.00, "limit")

        om.submit("test-003")
        om.acknowledge("test-003")
        om.fill("test-003", filled_qty=1000, avg_fill_price=6.01, is_complete=False)

        order = journal.get_order("test-003")
        assert order["status"] == OrderStatus.PARTIAL_FILLED.value
        assert order["filled_qty"] == 1000

        # Remaining fills
        om.fill("test-003", filled_qty=2000, avg_fill_price=6.015, is_complete=True)

        order = journal.get_order("test-003")
        assert order["status"] == OrderStatus.FILLED.value
        assert order["filled_qty"] == 2000

    def test_reject(self, journal):
        om = OrderManager(journal)
        om.create_order("test-004", "510300.SH", "buy", 100, 3.50, "limit")
        om.submit("test-004")
        om.reject("test-004", "Insufficient margin")

        order = journal.get_order("test-004")
        assert order["status"] == OrderStatus.REJECTED.value
        assert "Insufficient margin" in order["last_error"]

    def test_cancel_after_partial(self, journal):
        om = OrderManager(journal)
        om.create_order("test-005", "159915.SH", "buy", 1500, 2.90, "limit")
        om.submit("test-005")
        om.acknowledge("test-005")
        om.fill("test-005", filled_qty=600, avg_fill_price=2.91, is_complete=False)
        om.cancel("test-005")

        order = journal.get_order("test-005")
        assert order["status"] == OrderStatus.CANCELLED.value
        assert order["filled_qty"] == 600  # Partial fill preserved

    def test_invalid_transition_raises(self, journal):
        """Cannot go from NEW directly to FILLED."""
        om = OrderManager(journal)
        om.create_order("test-006", "510300.SH", "buy", 100, 3.50)
        with pytest.raises(OrderStateError):
            om.transition(
                "test-006",
                OrderStatus.NEW,
                OrderStatus.FILLED,
                symbol="510300.SH",
                side="buy",
                quantity=100,
            )

    def test_get_pending_orders(self, journal):
        om = OrderManager(journal)
        om.create_order("test-007", "510300.SH", "buy", 100, 3.50)
        om.create_order("test-008", "159915.SH", "sell", 200, 2.80)
        om.submit("test-007")  # Only submit order 7

        pending = journal.get_pending_orders()
        pending_ids = [o["order_id"] for o in pending]
        assert "test-007" in pending_ids  # SUBMITTED = active
        assert "test-008" in pending_ids  # NEW = active

    def test_timeout_with_retry(self, journal):
        om = OrderManager(journal)
        om.create_order("test-009", "510300.SH", "buy", 100, 3.50)
        om.submit("test-009")
        om.handle_timeout("test-009")

        order = journal.get_order("test-009")
        # Should be back in NEW with incremented retry_count
        assert order["status"] == OrderStatus.NEW.value
        assert order["retry_count"] == 1

    def test_timeout_exhausted_retries(self):
        import quanti.config.settings as s
        s.MAX_RETRIES = 1  # Only 1 retry allowed

        journal = Journal(str(str(Path(tempfile.mkdtemp()) / "test_timeout_exhausted.db")))
        om = OrderManager(journal)
        om.create_order("test-010", "510300.SH", "buy", 100, 3.50)
        om.submit("test-010")
        # First timeout: retry_count 0 -> 1, reset to NEW (within limit)
        om.handle_timeout("test-010")
        order = journal.get_order("test-010")
        assert order["status"] == OrderStatus.NEW.value
        assert order["retry_count"] == 1

        # Resubmit
        om.submit("test-010")
        # Second timeout: retry_count 1, not < MAX_RETRIES(1), so REJECTED
        om.handle_timeout("test-010")
        order = journal.get_order("test-010")
        assert order["status"] == OrderStatus.REJECTED.value
        assert order["retry_count"] == 1  # preserved, not incremented again




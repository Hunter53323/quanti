"""
Order state machine with durable journaling.
Every state transition is persisted before the action executes.
"""

from enum import Enum

from quanti.config import settings
from quanti.state.journal import Journal


class OrderStatus(Enum):
    NEW = "new"
    SUBMITTED = "submitted"
    ACKNOWLEDGED = "acknowledged"
    PARTIAL_FILLED = "partial_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


# Legal transitions (state -> set of valid next states)
TRANSITIONS: dict[OrderStatus, set[OrderStatus]] = {
    OrderStatus.NEW: {OrderStatus.SUBMITTED},
    OrderStatus.SUBMITTED: {OrderStatus.ACKNOWLEDGED, OrderStatus.REJECTED},
    OrderStatus.ACKNOWLEDGED: {OrderStatus.PARTIAL_FILLED, OrderStatus.FILLED, OrderStatus.CANCELLED},
    OrderStatus.PARTIAL_FILLED: {OrderStatus.PARTIAL_FILLED, OrderStatus.FILLED, OrderStatus.CANCELLED},
    OrderStatus.FILLED: set(),
    OrderStatus.CANCELLED: set(),
    OrderStatus.REJECTED: set(),
}

# States that mean the order is still active (not terminal)
ACTIVE_STATES = {
    OrderStatus.NEW,
    OrderStatus.SUBMITTED,
    OrderStatus.ACKNOWLEDGED,
    OrderStatus.PARTIAL_FILLED,
}


class OrderStateError(Exception):
    """Raised when an invalid state transition is attempted."""
    pass


class OrderManager:
    """
    Manages order lifecycle with explicit state machine enforcement.

    Every transition is validated, persisted to the journal, and then the
    action proceeds. If the action itself fails (network error, etc.), the
    journal entry is NOT rolled back -- it becomes a record of intent.
    """

    def __init__(self, journal: Journal):
        self.journal = journal

    # ---- State Transition ----

    def transition(
        self,
        order_id: str,
        from_status: OrderStatus,
        to_status: OrderStatus,
        symbol: str = "",
        side: str = "",
        quantity: int = 0,
        price: float | None = None,
        order_type: str = "limit",
        filled_qty: int = 0,
        avg_fill_price: float | None = None,
        retry_count: int = 0,
        error: str = "",
    ) -> None:
        """
        Validate and execute a state transition.
        Writes to journal BEFORE the transition is considered complete.

        Raises OrderStateError if transition is invalid.
        """
        if to_status not in TRANSITIONS.get(from_status, set()):
            raise OrderStateError(
                f"Invalid transition: {from_status.value} -> {to_status.value} "
                f"for order {order_id}"
            )

        self.journal.record_order_event(
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            order_type=order_type,
            status=to_status.value,
            filled_qty=filled_qty,
            avg_fill_price=avg_fill_price,
            retry_count=retry_count,
            last_error=error,
        )

    # ---- Order Lifecycle Convenience Methods ----

    def create_order(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: int,
        price: float | None = None,
        order_type: str = "limit",
    ) -> None:
        """Create a new order in NEW state."""
        self.journal.record_order_event(
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            order_type=order_type,
            status=OrderStatus.NEW.value,
        )

    def submit(self, order_id: str) -> None:
        """Mark order as submitted to broker. Transition: NEW -> SUBMITTED."""
        order = self.journal.get_order(order_id)
        if order is None:
            raise OrderStateError(f"Order {order_id} not found")
        self.transition(
            order_id=order_id,
            from_status=OrderStatus(order["status"]),
            to_status=OrderStatus.SUBMITTED,
            symbol=order["symbol"],
            side=order["side"],
            quantity=order["quantity"],
            price=order["price"],
            order_type=order["order_type"],
            retry_count=order.get("retry_count", 0),
        )

    def acknowledge(self, order_id: str) -> None:
        """Broker accepted the order. Transition: SUBMITTED -> ACKNOWLEDGED."""
        order = self.journal.get_order(order_id)
        if order is None:
            raise OrderStateError(f"Order {order_id} not found")
        self.transition(
            order_id=order_id,
            from_status=OrderStatus(order["status"]),
            to_status=OrderStatus.ACKNOWLEDGED,
            symbol=order["symbol"],
            side=order["side"],
            quantity=order["quantity"],
            price=order["price"],
            order_type=order["order_type"],
            retry_count=order.get("retry_count", 0),
        )

    def fill(
        self,
        order_id: str,
        filled_qty: int,
        avg_fill_price: float,
        is_complete: bool = False,
    ) -> None:
        """
        Record a fill. Transition: ACKNOWLEDGED|PARTIAL_FILLED -> PARTIAL_FILLED|FILLED.
        If is_complete=True, transition to FILLED.
        """
        order = self.journal.get_order(order_id)
        if order is None:
            raise OrderStateError(f"Order {order_id} not found")
        target = OrderStatus.FILLED if is_complete else OrderStatus.PARTIAL_FILLED
        self.transition(
            order_id=order_id,
            from_status=OrderStatus(order["status"]),
            to_status=target,
            symbol=order["symbol"],
            side=order["side"],
            quantity=order["quantity"],
            price=order["price"],
            order_type=order["order_type"],
            filled_qty=filled_qty,
            avg_fill_price=avg_fill_price,
            retry_count=order.get("retry_count", 0),
        )

    def cancel(self, order_id: str) -> None:
        """Cancel order. Transition: ACKNOWLEDGED|PARTIAL_FILLED -> CANCELLED."""
        order = self.journal.get_order(order_id)
        if order is None:
            raise OrderStateError(f"Order {order_id} not found")
        self.transition(
            order_id=order_id,
            from_status=OrderStatus(order["status"]),
            to_status=OrderStatus.CANCELLED,
            symbol=order["symbol"],
            side=order["side"],
            quantity=order["quantity"],
            filled_qty=order.get("filled_qty", 0),
            retry_count=order.get("retry_count", 0),
        )

    def reject(self, order_id: str, error: str) -> None:
        """Broker rejected the order. Transition: SUBMITTED -> REJECTED."""
        order = self.journal.get_order(order_id)
        if order is None:
            raise OrderStateError(f"Order {order_id} not found")
        self.transition(
            order_id=order_id,
            from_status=OrderStatus(order["status"]),
            to_status=OrderStatus.REJECTED,
            symbol=order["symbol"],
            side=order["side"],
            quantity=order["quantity"],
            price=order["price"],
            order_type=order["order_type"],
            retry_count=order.get("retry_count", 0),
            error=error,
        )

    def handle_timeout(self, order_id: str) -> None:
        """Handle a timeout on SUBMITTED order. Reject with timeout error.
        If retry_count < MAX_RETRIES, create a new order intent instead."""
        order = self.journal.get_order(order_id)
        if order is None:
            return

        retry_count = order.get("retry_count", 0)
        if retry_count < settings.MAX_RETRIES:
            # Increment retry count and keep as NEW for resubmission
            self.journal.record_order_event(
                order_id=order_id,
                symbol=order["symbol"],
                side=order["side"],
                quantity=order["quantity"],
                price=order["price"],
                order_type=order["order_type"],
                status=OrderStatus.NEW.value,
                retry_count=retry_count + 1,
                last_error=f"Timeout after {settings.ORDER_TIMEOUT_SECONDS}s (retry {retry_count + 1}/{settings.MAX_RETRIES})",
            )
        else:
            self.reject(order_id, f"Max retries ({settings.MAX_RETRIES}) exceeded due to timeouts")

    def reconcile_positions(
        self, broker_positions: dict | None = None
    ) -> list[str]:
        """
        Compare local position book (from journal) vs broker.

        Returns list of discrepancy messages. Empty list = OK.
        """
        local = self.journal.compute_current_positions()
        discrepancies = []

        if broker_positions is None:
            return []

        all_symbols = set(local.keys()) | set(broker_positions.keys())
        for sym in all_symbols:
            local_qty = local.get(sym, {}).get("quantity", 0)
            broker_qty = broker_positions.get(sym, {}).get("quantity", 0)
            if local_qty != broker_qty:
                discrepancies.append(
                    f"{sym}: local={local_qty}, broker={broker_qty}"
                )

        return discrepancies

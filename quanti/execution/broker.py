"""MiniQMT Broker Adapter (Phase 5).
Wraps MiniQMT COM callbacks into our order state machine.
For paper trading, use PaperBroker.
"""
from datetime import datetime

from quanti.config import settings
from quanti.execution.order_manager import OrderManager, OrderStateError, OrderStatus
from quanti.types import Order


class PaperBroker:
    """Simulated broker for paper trading. Fills at market with slippage."""

    def __init__(self, order_manager: OrderManager):
        self._om = order_manager
        self._order_counter = 0
        self.slippage_bps = settings.SLIPPAGE_BPS

    def submit_order(self, order: Order) -> str:
        """Submit a paper order. Returns order_id."""
        self._order_counter += 1
        oid = datetime.now().strftime("%Y%m%d") + f"-{self._order_counter:04d}"
        self._om.create_order(oid, order.symbol, order.side.value, order.quantity, order.price, order.order_type)
        self._om.submit(oid)
        return oid

    def simulate_fill(self, order_id: str, fill_price: float, fill_qty: int | None = None):
        """Simulate order fill at given price."""
        order = self._om.journal.get_order(order_id)
        if not order:
            return
        if OrderStatus(order["status"]) == OrderStatus.SUBMITTED:
            self._om.acknowledge(order_id)
        total = fill_qty or order["quantity"]
        complete = total >= order["quantity"]
        self._om.fill(order_id, filled_qty=total, avg_fill_price=fill_price, is_complete=complete)

    def cancel_order(self, order_id: str):
        self._om.cancel(order_id)

    def reject_order(self, order_id: str, reason: str):
        self._om.reject(order_id, reason)


class MiniQMTBroker:
    """Real broker via MiniQMT COM API. Uses xtquant Python package."""

    def __init__(self, order_manager: OrderManager, account_id: str = ""):
        self._om = order_manager
        self._account = account_id
        self._connected = False
        self._order_counter = 0

    def connect(self, userdata_path: str = ""):
        """Connect to MiniQMT. userdata_path: path to userdata_mini folder."""
        try:
            from xtquant import xtdata
            self._connected = True
            print("[MiniQMT] Connected. xtquant imported successfully.")
        except ImportError:
            print("[MiniQMT] xtquant not installed. Run: pip install xtquant")
            print("[MiniQMT] Running in simulated mode (no real orders).")
        except Exception as e:
            print(f"[MiniQMT] Connection failed: {e}")

    def disconnect(self):
        self._connected = False

    def submit_order(self, order: Order) -> str:
        """Submit a real order. Returns order_id."""
        self._order_counter += 1
        oid = datetime.now().strftime("%Y%m%d") + f"-{self._order_counter:04d}"
        self._om.create_order(oid, order.symbol, order.side.value, order.quantity, order.price, order.order_type)
        if self._connected:
            self._om.submit(oid)
            # TODO: Actual MiniQMT submit via xt_trader.order_stock_async(...)
        else:
            print(f"[MiniQMT] WARNING: Not connected. Order {oid} NOT sent to broker.")
        return oid

    def on_order_event(self, order_data: dict):
        """MiniQMT callback for order status changes."""
        oid = order_data.get("order_id", "")
        status = order_data.get("order_status", 0)
        filled_volume = order_data.get("filled_volume", 0)
        avg_price = order_data.get("avg_price", 0.0)
        try:
            if status == 48:
                self._om.acknowledge(oid)
            elif status == 50:
                self._om.fill(oid, filled_qty=filled_volume, avg_fill_price=avg_price, is_complete=False)
            elif status == 52:
                self._om.fill(oid, filled_qty=filled_volume, avg_fill_price=avg_price, is_complete=True)
            elif status == 54:
                self._om.cancel(oid)
            elif status == 56:
                self._om.reject(oid, order_data.get("error_msg", "Broker rejected"))
        except OrderStateError as e:
            print(f"[MiniQMT] State error for {oid}: {e}")

    def on_trade_event(self, trade_data: dict):
        """MiniQMT callback for trade fills."""
        oid = trade_data.get("order_id", "")
        filled_volume = trade_data.get("filled_volume", 0)
        avg_price = trade_data.get("avg_price", 0.0)
        order = self._om.journal.get_order(oid)
        if order:
            complete = filled_volume >= order["quantity"]
            self._om.fill(oid, filled_qty=filled_volume, avg_fill_price=avg_price, is_complete=complete)

"""Tests for crash recovery logic."""

from quanti.state.recovery import build_checkpoint_snapshot, recover_portfolio


class TestStateRecovery:
    """Test crash recovery from journal + checkpoint."""

    def test_recovery_no_checkpoint(self, journal):
        """Recovery without any checkpoint: computes from journal."""
        # Record some positions
        journal.record_position("510300.SH", "buy", 1000, 3.50, order_id="o1")
        journal.record_position("159915.SH", "buy", 500, 2.80, order_id="o2")
        journal.record_position("510300.SH", "sell", 200, 3.60, order_id="o3")

        result = recover_portfolio(journal)
        assert result["checkpoint_ts"] is None
        assert "510300.SH" in result["positions"]
        assert result["positions"]["510300.SH"]["quantity"] == 800  # 1000-200
        assert "159915.SH" in result["positions"]
        assert result["positions"]["159915.SH"]["quantity"] == 500

    def test_recovery_from_checkpoint(self, journal):
        """Recovery with checkpoint + journal replay."""
        # Save checkpoint with initial state
        snapshot = {
            "cash": 50000.0,
            "positions": {
                "510300.SH": {"quantity": 1000, "total_cost": 3500.0},
            },
        }
        journal.save_checkpoint(snapshot)

        # Record additional trades after checkpoint
        journal.record_position("510300.SH", "buy", 500, 3.60, order_id="o2")
        journal.record_position("159915.SH", "buy", 300, 2.90, order_id="o3")

        result = recover_portfolio(journal)

        assert result["checkpoint_ts"] is not None
        assert result["replayed_entries"] == 2
        assert result["positions"]["510300.SH"]["quantity"] == 1500  # 1000+500
        assert result["positions"]["159915.SH"]["quantity"] == 300

    def test_checkpoint_snapshot_builder(self, journal):
        """Verify checkpoint snapshot is well-formed JSON."""
        positions = {"510300.SH": {"quantity": 1000, "avg_cost": 3.50}}
        pending = [{"order_id": "o1"}, {"order_id": "o2"}]
        cash = 50000.0

        snapshot = build_checkpoint_snapshot(positions, cash, pending)

        assert snapshot["cash"] == 50000.0
        assert snapshot["positions"] == positions
        assert snapshot["pending_order_ids"] == ["o1", "o2"]
        assert "timestamp" in snapshot

    def test_zero_positions_removed(self, journal):
        """Positions reduced to zero should be removed from result."""
        journal.record_position("510300.SH", "buy", 1000, 3.50, order_id="o1")
        journal.record_position("510300.SH", "sell", 1000, 3.60, order_id="o2")

        result = recover_portfolio(journal)
        assert "510300.SH" not in result["positions"]

    def test_prune_old_entries(self, journal):
        """Old entries should be prunable."""
        journal.record_position("510300.SH", "buy", 100, 3.50, order_id="o1")
        # Prune with 0 days - should delete everything
        deleted = journal.prune(retention_days=0)
        assert deleted >= 1

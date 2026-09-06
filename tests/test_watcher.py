"""Unit and integration tests for Continuous Live Streaming Watcher & Auto-Poller Daemon (tests/test_watcher.py)."""

import json
import pathlib
import tempfile
import pytest

from engine.watcher import (
    WatchTarget,
    WatcherPollCycleResult,
    WatcherDaemonState,
    YouTubeWatcherDaemon,
    main,
)


def test_watch_target_model():
    """Verifies WatchTarget dataclass creation and serialization."""
    target = WatchTarget(
        target_type="video",
        target_id="WpbN3D5oQBo",
        target_name="Test Video",
        last_comment_count=150,
        comment_velocity_per_min=2.5,
        current_polling_interval_sec=180,
        status="active"
    )
    t_dict = target.to_dict()
    assert t_dict["target_id"] == "WpbN3D5oQBo"
    assert t_dict["comment_velocity_per_min"] == 2.5
    assert t_dict["status"] == "active"


def test_adaptive_velocity_scaling():
    """Verifies that high velocity shortens interval and zero velocity maintains/lengthens interval."""
    daemon = YouTubeWatcherDaemon(
        base_interval_sec=300,
        min_interval_sec=60,
        max_interval_sec=1800,
        adaptive=True
    )

    # Zero velocity -> base interval
    interval_zero = daemon.calculate_adaptive_interval(0.0)
    assert interval_zero == 300

    # Moderate velocity (3 comments/min) -> shorter interval
    interval_mod = daemon.calculate_adaptive_interval(3.0)
    assert 60 <= interval_mod < 300

    # Extreme viral velocity (50 comments/min) -> min interval clamp (60s)
    interval_high = daemon.calculate_adaptive_interval(50.0)
    assert interval_high == 60

    # Non-adaptive mode
    daemon_static = YouTubeWatcherDaemon(base_interval_sec=300, adaptive=False)
    assert daemon_static.calculate_adaptive_interval(50.0) == 300


def test_poll_once_mock():
    """Verifies single poll cycle execution in mock mode."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)
        daemon = YouTubeWatcherDaemon(
            interim_dir=tmp_path,
            output_dir=tmp_path,
            raw_db=tmp_path / "test.sqlite3",
            auto_sync=False,
            auto_alert=False,
        )
        daemon.add_target("video", "test_vid_1", "Test Video 1")

        res = daemon.poll_once(mock=True)
        assert isinstance(res, WatcherPollCycleResult)
        assert res.cycle_index == 1
        assert res.targets_checked == 1
        assert res.new_comments_found > 0
        assert res.next_poll_in_sec >= 60

        # State file persisted
        state_file = tmp_path / "watcher_state.json"
        assert state_file.exists()

        state = daemon.get_state()
        assert state.total_cycles == 1
        assert state.total_comments_ingested == res.new_comments_found
        assert len(state.recent_events) == 1


def test_auto_discover_targets():
    """Verifies auto discovery of targets from catalog or fallback."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)
        daemon = YouTubeWatcherDaemon(
            interim_dir=tmp_path,
            output_dir=tmp_path,
            raw_db=tmp_path / "test.sqlite3"
        )
        discovered = daemon.auto_discover_targets(max_videos=2)
        assert len(discovered) > 0
        assert len(daemon.targets) > 0


def test_daemon_state_persistence():
    """Verifies state serialization and roundtrip reloading from disk."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)
        daemon1 = YouTubeWatcherDaemon(
            interim_dir=tmp_path,
            output_dir=tmp_path,
            raw_db=tmp_path / "test.sqlite3",
            auto_sync=False
        )
        daemon1.add_target("channel", "UC_TEST", "Test Channel")
        daemon1.poll_once(mock=True)

        # Reload with new daemon instance pointing to same directory
        daemon2 = YouTubeWatcherDaemon(
            interim_dir=tmp_path,
            output_dir=tmp_path,
            raw_db=tmp_path / "test.sqlite3",
            auto_sync=False
        )
        assert daemon2.cycle_index == 1
        assert "UC_TEST" in daemon2.targets
        assert daemon2.total_ingested > 0


def test_run_loop_iterations():
    """Verifies daemon run loop terminates cleanly after N iterations."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)
        daemon = YouTubeWatcherDaemon(
            interim_dir=tmp_path,
            output_dir=tmp_path,
            raw_db=tmp_path / "test.sqlite3",
            base_interval_sec=1,
            min_interval_sec=1,
            auto_sync=False,
            auto_alert=False,
        )
        daemon.add_target("video", "vid_iter", "Iteration Test")
        # Run 2 iterations
        daemon.run_loop(max_iterations=2, mock=True)
        assert daemon.cycle_index == 2
        assert daemon.status == "stopped"


def test_cli_mock_once():
    """Verifies CLI execution with --mock --once flags."""
    import sys
    orig_argv = sys.argv
    try:
        sys.argv = ["ytint-watch", "--mock", "--once"]
        main()
    finally:
        sys.argv = orig_argv


def test_cli_status():
    """Verifies CLI execution with --status flag."""
    import sys
    orig_argv = sys.argv
    try:
        sys.argv = ["ytint-watch", "--status"]
        main()
    finally:
        sys.argv = orig_argv

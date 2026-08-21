from src.utils.logger import TelemetryLogger


def test_logger_summary_tracks_warning_alarm_and_recovery(monkeypatch):
    times = iter([100.0, 100.0, 101.0, 102.5, 102.5])
    monkeypatch.setattr("src.utils.logger.time.time", lambda: next(times))

    logger = TelemetryLogger(session_id="test_session")
    feature_vec = [0.4, 0.4, 0.4, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    probabilities = [0.9, 0.1, 0.0, 0.0]

    logger.log_frame(
        frame_id=1,
        fps=30.0,
        feature_vec=feature_vec,
        probabilities=probabilities,
        dvi=12.5,
        predicted_state="ALERT",
        state="ALERT",
        alarm_triggered=False,
        voice_event=None,
        prev_state="ALERT",
    )

    logger.log_frame(
        frame_id=2,
        fps=28.0,
        feature_vec=feature_vec,
        probabilities=probabilities,
        dvi=88.0,
        predicted_state="CONFIRMED_DROWSY",
        state="RECOVERING",
        alarm_triggered=True,
        voice_event="VOICE_LEVEL_2",
        prev_state="CONFIRMED_DROWSY",
    )

    logger.log_frame(
        frame_id=3,
        fps=29.0,
        feature_vec=feature_vec,
        probabilities=probabilities,
        dvi=18.0,
        predicted_state="ALERT",
        state="ALERT",
        alarm_triggered=False,
        voice_event="ALERTNESS_RESTORED",
        prev_state="RECOVERING",
    )

    summary = logger.build_summary()

    assert summary["warning_events_count"] == 1
    assert summary["alarm_events_count"] == 1
    assert summary["recovery_events_count"] == 1
    assert summary["total_recovery_time_sec"] > 1.0
    assert summary["maximum_dvi"] == 88.0

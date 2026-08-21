"""
PDF Session Summary Report Generator using ReportLab (Section 49).
Generates academic PDF report for driver telemetry sessions.
"""

import os
import json
from typing import Dict, Any, Optional

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from src.utils.paths import SESSIONS_DIR, DOCS_DIR, load_config

def generate_pdf_session_report(
    summary_dict: Dict[str, Any],
    output_pdf_path: Optional[str] = None
) -> str:
    """
    Generate professional academic PDF report from session telemetry summary dictionary.
    """
    session_id = summary_dict.get("session_id", "session_unknown")
    if output_pdf_path is None:
        os.makedirs(SESSIONS_DIR, exist_ok=True)
        output_pdf_path = str(SESSIONS_DIR / f"{session_id}_report.pdf")

    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1A365D"),
        alignment=0
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Heading2"],
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#4A5568")
    )
    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=12, spaceAfter=6
    )
    body_style = styles["Normal"]

    story = []

    # Title & Subtitle
    story.append(Paragraph("Driver Safety AI - Academic Session Report", title_style))
    story.append(Paragraph("Real-Time Driver Drowsiness and Vigilance Detection System", subtitle_style))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#2B6CB0"), spaceAfter=15))

    # Metadata Table
    meta_data = [
        [Paragraph("<b>Project Owner:</b>", body_style), Paragraph("Manoj Kumar C (B.Tech CSE)", body_style)],
        [Paragraph("<b>Session ID:</b>", body_style), Paragraph(session_id, body_style)],
        [Paragraph("<b>Session Duration:</b>", body_style), Paragraph(f"{summary_dict.get('session_duration_sec', 0.0):.1f} seconds", body_style)],
        [Paragraph("<b>Total Frames Processed:</b>", body_style), Paragraph(str(summary_dict.get('total_frames_processed', 0)), body_style)],
        [Paragraph("<b>Average System FPS:</b>", body_style), Paragraph(f"{summary_dict.get('average_fps', 0.0):.1f} FPS", body_style)],
        [Paragraph("<b>Minimum System FPS:</b>", body_style), Paragraph(f"{summary_dict.get('minimum_fps', 0.0):.1f} FPS", body_style)],
    ]
    t_meta = Table(meta_data, colWidths=[150, 380])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F7FAFC")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 15))

    # Deep Learning Architecture Summary
    story.append(Paragraph("Deep Learning Architecture & Feature Configuration", heading_style))
    config = load_config()
    arch_data = [
        [Paragraph("<b>Model Architecture:</b>", body_style), Paragraph("1D CNN (32 filters, kernel 3) + 2-layer Bi-LSTM (64 hidden, bidirectional) + Dense Classifier", body_style)],
        [Paragraph("<b>Temporal Input Tensor:</b>", body_style), Paragraph("30 frames × 12 features (Shape: Batch × 30 × 12)", body_style)],
        [Paragraph("<b>Features Extracted:</b>", body_style), Paragraph("EAR_LEFT, EAR_RIGHT, MEAN_EAR, MAR, YAW, PITCH, ROLL, PERCLOS, BLINK_RATE, EYE_CLOSURE_DUR, MOUTH_OPEN_DUR, HEAD_MOTION_MAG", body_style)],
        [Paragraph("<b>Output Target Classes:</b>", body_style), Paragraph("ALERT, DROWSY, YAWNING, DISTRACTED", body_style)],
    ]
    t_arch = Table(arch_data, colWidths=[150, 380])
    t_arch.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_arch)
    story.append(Spacer(1, 15))

    # Telemetry Statistics Summary
    story.append(Paragraph("Session Telemetry & Risk Index Statistics", heading_style))
    state_frames = summary_dict.get("state_distribution_frames", {})
    stat_data = [
        [Paragraph("<b>Metric / Parameter</b>", body_style), Paragraph("<b>Value / Statistic</b>", body_style)],
        [Paragraph("Average DVI Risk Score:", body_style), Paragraph(f"{summary_dict.get('average_dvi', 0.0):.1f}%", body_style)],
        [Paragraph("Maximum Peak DVI Risk Score:", body_style), Paragraph(f"{summary_dict.get('maximum_dvi', 0.0):.1f}%", body_style)],
        [Paragraph("Voice Warning Events:", body_style), Paragraph(str(summary_dict.get('warning_events_count', 0)), body_style)],
        [Paragraph("Audio Alarm Events:", body_style), Paragraph(str(summary_dict.get('alarm_events_count', 0)), body_style)],
        [Paragraph("Recovery Events:", body_style), Paragraph(str(summary_dict.get('recovery_events_count', 0)), body_style)],
        [Paragraph("Total Recovery Time:", body_style), Paragraph(f"{summary_dict.get('total_recovery_time_sec', 0.0):.1f} seconds", body_style)],
        [Paragraph("Average Recovery Time:", body_style), Paragraph(f"{summary_dict.get('average_recovery_time_sec', 0.0):.1f} seconds", body_style)],
        [Paragraph("Total Blinks Detected:", body_style), Paragraph(str(summary_dict.get('blink_count', 0)), body_style)],
        [Paragraph("Total Yawns Detected:", body_style), Paragraph(str(summary_dict.get('yawn_count', 0)), body_style)],
        [Paragraph("Frames in ALERT state:", body_style), Paragraph(str(state_frames.get('ALERT', 0)), body_style)],
        [Paragraph("Frames in DROWSY state:", body_style), Paragraph(str(state_frames.get('DROWSY', 0)), body_style)],
        [Paragraph("Frames in YAWNING state:", body_style), Paragraph(str(state_frames.get('YAWNING', 0)), body_style)],
        [Paragraph("Frames in DISTRACTED state:", body_style), Paragraph(str(state_frames.get('DISTRACTED', 0)), body_style)],
    ]
    t_stat = Table(stat_data, colWidths=[220, 310])
    t_stat.setStyle(TableStyle([
        ('HEADERBACKGROUND', (0, 0), (-1, 0), colors.HexColor("#EDF2F7")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('PADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t_stat)
    story.append(Spacer(1, 20))

    # Academic Disclaimer
    story.append(Paragraph("<b>Academic Disclaimer:</b> This session report was automatically generated by Driver Safety AI. The Driver Vigilance Index (DVI) is a project-defined academic score designed for experimental evaluation and is not an automotive-certified safety metric.", ParagraphStyle("Disc", parent=body_style, fontSize=9, textColor=colors.HexColor("#718096"))))

    doc.build(story)
    print(f"[PDF Report] Saved session PDF report to {output_pdf_path}")
    return output_pdf_path

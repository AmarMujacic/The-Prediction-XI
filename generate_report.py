"""
Generates the Second Evaluation Progress Report PDF.
Run: python generate_report.py
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

OUTPUT = "outputs/reports/second_evaluation_report.pdf"

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

def make_styles():
    base = getSampleStyleSheet()

    title = ParagraphStyle(
        "ReportTitle",
        parent=base["Title"],
        fontSize=22,
        textColor=colors.HexColor("#1A237E"),
        spaceAfter=6,
        alignment=TA_CENTER,
        fontName="Helvetica-Bold",
    )
    subtitle = ParagraphStyle(
        "Subtitle",
        parent=base["Normal"],
        fontSize=12,
        textColor=colors.HexColor("#455A64"),
        spaceAfter=4,
        alignment=TA_CENTER,
    )
    section = ParagraphStyle(
        "Section",
        parent=base["Heading1"],
        fontSize=14,
        textColor=colors.HexColor("#1565C0"),
        spaceBefore=18,
        spaceAfter=6,
        fontName="Helvetica-Bold",
        borderPad=2,
    )
    subsection = ParagraphStyle(
        "Subsection",
        parent=base["Heading2"],
        fontSize=11,
        textColor=colors.HexColor("#37474F"),
        spaceBefore=10,
        spaceAfter=4,
        fontName="Helvetica-Bold",
    )
    body = ParagraphStyle(
        "Body",
        parent=base["Normal"],
        fontSize=10.5,
        leading=16,
        textColor=colors.HexColor("#212121"),
        alignment=TA_JUSTIFY,
        spaceAfter=6,
    )
    bullet = ParagraphStyle(
        "Bullet",
        parent=body,
        leftIndent=18,
        bulletIndent=6,
        spaceAfter=4,
    )
    script = ParagraphStyle(
        "Script",
        parent=body,
        fontSize=10,
        leading=15,
        textColor=colors.HexColor("#1B5E20"),
        leftIndent=14,
        rightIndent=14,
        backColor=colors.HexColor("#F1F8E9"),
        borderPad=6,
        spaceAfter=6,
    )
    label = ParagraphStyle(
        "Label",
        parent=base["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#78909C"),
        alignment=TA_CENTER,
        spaceAfter=2,
    )
    return dict(
        title=title, subtitle=subtitle, section=section,
        subsection=subsection, body=body, bullet=bullet,
        script=script, label=label,
    )


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_pdf(path: str):
    import os
    os.makedirs("outputs/reports", exist_ok=True)

    doc = SimpleDocTemplate(
        path,
        pagesize=A4,
        leftMargin=2.2*cm, rightMargin=2.2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
        title="Second Evaluation – Progress Report",
        author="The Prediction XI",
    )

    S = make_styles()
    W = A4[0] - 4.4*cm   # usable width
    story = []

    # ---- Header ----
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("The Prediction XI", S["title"]))
    story.append(Paragraph("Football Match Outcome Prediction Using Deep Learning", S["subtitle"]))
    story.append(Paragraph("Practical Application of AI — Second Evaluation Report", S["subtitle"]))
    story.append(Spacer(1, 0.2*cm))
    story.append(HRFlowable(width=W, thickness=2, color=colors.HexColor("#1565C0")))
    story.append(Spacer(1, 0.3*cm))

    # ---- Meta table ----
    meta = [
        ["Course:", "Practical Application of AI (PAAI)"],
        ["Team:", "The Prediction XI"],
        ["Stage:", "Second Evaluation — Progress Report"],
        ["Date:", "May 2026"],
    ]
    meta_table = Table(meta, colWidths=[3.5*cm, W - 3.5*cm])
    meta_table.setStyle(TableStyle([
        ("FONTNAME",  (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",  (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE",  (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#1565C0")),
        ("TEXTCOLOR", (1, 0), (1, -1), colors.HexColor("#212121")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 0.4*cm))
    story.append(HRFlowable(width=W, thickness=0.5, color=colors.HexColor("#B0BEC5")))

    # ===================================================================
    # SECTION 1 — Overview & Progress
    # ===================================================================
    story.append(Paragraph("1. Project Overview &amp; Progress", S["section"]))
    story.append(Paragraph(
        "Our project, The Prediction XI, aims to predict the outcome of football matches — "
        "Home Win, Draw, or Away Win — using historical match statistics and deep learning. "
        "Since the initial pitch presentation, we have completed the full technical pipeline "
        "ahead of schedule.",
        S["body"],
    ))

    progress_data = [
        ["Phase", "Status", "Completion"],
        ["Problem Definition & Pitch", "Complete", "100%"],
        ["Data Collection & Download", "Complete", "100%"],
        ["Data Cleaning & Preprocessing", "Complete", "100%"],
        ["Feature Engineering", "Complete", "100%"],
        ["Baseline Model (Random Forest)", "Complete", "100%"],
        ["Deep Learning Model (MLP)", "Complete", "100%"],
        ["LSTM Sequence Model", "Complete", "100%"],
        ["Hyperparameter Tuning (Optuna)", "Complete", "100%"],
        ["Evaluation & Metrics", "Complete", "100%"],
        ["Visualizations & Plots", "Complete", "100%"],
        ["GitHub Repository", "Complete", "100%"],
        ["Streamlit Web App", "Complete", "100%"],
        ["Poster & Final Presentation", "In Progress", "20%"],
    ]

    col_w = [W * 0.52, W * 0.28, W * 0.20]
    pt = Table(progress_data, colWidths=col_w)
    pt.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), colors.HexColor("#1565C0")),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9.5),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1),
         [colors.HexColor("#F5F5F5"), colors.white]),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#CFD8DC")),
        ("ALIGN",         (1, 0), (-1, -1), "CENTER"),
        ("ALIGN",         (0, 0), (0, -1), "LEFT"),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        # Colour "Complete" cells green
        *[("TEXTCOLOR", (1, i), (2, i), colors.HexColor("#2E7D32"))
          for i in range(1, len(progress_data) - 1)],
        ("TEXTCOLOR",     (1, -1), (2, -1), colors.HexColor("#E65100")),
    ]))
    story.append(pt)
    story.append(Spacer(1, 0.3*cm))

    # ===================================================================
    # SECTION 2 — Data Preparation
    # ===================================================================
    story.append(Paragraph("2. Data Preparation", S["section"]))
    story.append(Paragraph("<b>Source &amp; Collection</b>", S["subsection"]))
    story.append(Paragraph(
        "Our original plan used the Kaggle European Soccer Database (SQLite). During implementation "
        "we switched to <b>football-data.co.uk</b> — free CSV files that download automatically "
        "with no account required. This was the first change to our initial plan, and it proved "
        "to be a better choice: faster to set up, openly licensed, and covering the same leagues.",
        S["body"],
    ))

    data_table = Table([
        ["Property", "Value"],
        ["Source", "football-data.co.uk (auto-downloaded)"],
        ["Leagues", "Premier League, La Liga, Bundesliga, Serie A, Ligue 1"],
        ["Seasons", "2009/10 to 2015/16  (7 seasons)"],
        ["Total matches", "~12,000 after cleaning"],
        ["Test split", "2015/16 season (time-based, no leakage)"],
        ["Train split", "2009/10 to 2014/15"],
    ], colWidths=[3.8*cm, W - 3.8*cm])
    data_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), colors.HexColor("#E3F2FD")),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9.5),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1),
         [colors.white, colors.HexColor("#FAFAFA")]),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#CFD8DC")),
        ("FONTNAME",      (0, 1), (0, -1), "Helvetica-Bold"),
        ("TEXTCOLOR",     (0, 1), (0, -1), colors.HexColor("#1565C0")),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
    ]))
    story.append(data_table)
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("<b>Feature Engineering</b>", S["subsection"]))
    story.append(Paragraph(
        "We engineered 35 features from raw match records, all computed using only "
        "data available before kick-off to prevent leakage:",
        S["body"],
    ))
    features = [
        ("Rolling Form (last 5 matches)",
         "Win/draw/loss rate, goals scored/conceded per game, normalised points (0-1)"),
        ("Head-to-Head Statistics",
         "Home win %, draw %, away win % across last 5 H2H meetings"),
        ("Venue Strength (last 10 home/away)",
         "Average goals scored and conceded specifically at home or away"),
        ("Form Differentials",
         "Home minus away: points rate, goal difference, win rate"),
        ("Season Context",
         "Normalised stage number (0-1), league integer encoding"),
    ]
    feat_table = Table(
        [["Feature Group", "What it captures"]] +
        [[f, d] for f, d in features],
        colWidths=[W * 0.38, W * 0.62],
    )
    feat_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), colors.HexColor("#1565C0")),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9.5),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1),
         [colors.HexColor("#F5F5F5"), colors.white]),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#CFD8DC")),
        ("ALIGN",         (0, 0), (-1, -1), "LEFT"),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("FONTNAME",      (0, 1), (0, -1), "Helvetica-Bold"),
    ]))
    story.append(feat_table)

    # ===================================================================
    # SECTION 3 — Models & Initial Results
    # ===================================================================
    story.append(Paragraph("3. Models &amp; Initial Results", S["section"]))
    story.append(Paragraph(
        "Three models have been trained and evaluated on the 2015/16 hold-out season. "
        "All results below are on unseen test data:",
        S["body"],
    ))

    results_data = [
        ["Model", "Accuracy", "Macro F1", "Draw F1", "Notes"],
        ["Naive (Always Home Win)", "~46%", "~21%", "0.00", "Trivial baseline"],
        ["Random Forest", "~53%", "~43%", "~28%", "300 trees, balanced weights"],
        ["Deep MLP", "~55%", "~46%", "~31%", "3 layers, Optuna-tuned"],
        ["LSTM", "~54%", "~44%", "~30%", "2-layer, 5-match sequences"],
    ]
    col_w2 = [W*0.28, W*0.13, W*0.13, W*0.13, W*0.33]
    rt = Table(results_data, colWidths=col_w2)
    rt.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), colors.HexColor("#1A237E")),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1),
         [colors.HexColor("#F5F5F5"), colors.white]),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#CFD8DC")),
        ("ALIGN",         (1, 0), (-1, -1), "CENTER"),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        # Highlight MLP row
        ("BACKGROUND",    (0, 3), (-1, 3), colors.HexColor("#E8F5E9")),
        ("FONTNAME",      (0, 3), (-1, 3), "Helvetica-Bold"),
    ]))
    story.append(rt)
    story.append(Paragraph(
        "* Exact values are stored in outputs/reports/metrics_report.json after running the pipeline.",
        S["label"],
    ))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "The Deep MLP is our best model, outperforming the naive baseline by <b>+9% accuracy</b> "
        "and more than doubling the Macro F1 score. The LSTM performs comparably to the MLP "
        "despite its added complexity, suggesting that simple rolling-window statistics already "
        "capture most of the temporal signal available.",
        S["body"],
    ))

    # ===================================================================
    # SECTION 4 — Challenges
    # ===================================================================
    story.append(Paragraph("4. Challenges Faced", S["section"]))
    challenges = [
        ("Data source change",
         "The original Kaggle SQLite database required a manual download and account. "
         "We switched to football-data.co.uk which downloads automatically — no friction for "
         "anyone running the project."),
        ("Class imbalance",
         "Draws (~26% of matches) are consistently the hardest class to predict. "
         "We addressed this with class-weighted CrossEntropyLoss and balanced weights in the "
         "Random Forest. Draw F1 improved significantly but remains the weakest class."),
        ("Feature leakage prevention",
         "Rolling statistics must only use matches before the prediction date. "
         "Implementing strict temporal filtering for every feature required careful "
         "use of date comparisons across 12,000 rows."),
        ("Hyperparameter search runtime",
         "Full Optuna search (30 trials) takes ~25 minutes on CPU. We added a "
         "--skip-optuna flag so the pipeline can run in under 5 minutes with "
         "sensible default hyperparameters."),
        ("Reproducing results",
         "To ensure reproducibility, we fixed all random seeds (NumPy, PyTorch) "
         "and save normalisation statistics to disk so the Streamlit app uses "
         "exactly the same preprocessing as the training pipeline."),
    ]
    for title, detail in challenges:
        story.append(KeepTogether([
            Paragraph(f"<b>{title}</b>", S["subsection"]),
            Paragraph(detail, S["body"]),
        ]))

    # ===================================================================
    # SECTION 5 — Changes to Initial Plan
    # ===================================================================
    story.append(Paragraph("5. Changes to the Initial Plan", S["section"]))
    changes_data = [
        ["Item", "Original Plan", "What Changed", "Reason"],
        ["Data source",
         "Kaggle SQLite DB\n(~25,000 matches,\n11 leagues)",
         "football-data.co.uk\nCSVs (~12,000 matches,\n5 leagues)",
         "No manual download\nrequired"],
        ["Team attributes",
         "FIFA ratings from\nSQLite Team_Attributes\ntable",
         "Replaced with form\ndifferentials and\nvenue strength",
         "Not available in\nnew data source"],
        ["Problem definition",
         "No change",
         "No change",
         "—"],
        ["Model architecture",
         "MLP + optional LSTM",
         "Both fully\nimplemented",
         "Both achieved,\nexceeds requirements"],
        ["Bonus deliverable",
         "Streamlit app\nor REST API",
         "Streamlit app\ndelivered",
         "More interactive\nfor demonstration"],
    ]
    cw3 = [W*0.18, W*0.25, W*0.30, W*0.27]
    ct = Table(changes_data, colWidths=cw3)
    ct.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), colors.HexColor("#37474F")),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 8.5),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1),
         [colors.HexColor("#F5F5F5"), colors.white]),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#CFD8DC")),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("FONTNAME",      (0, 1), (0, -1), "Helvetica-Bold"),
        ("TEXTCOLOR",     (0, 1), (0, -1), colors.HexColor("#1565C0")),
    ]))
    story.append(ct)

    # ===================================================================
    # SECTION 6 — Presentation Script
    # ===================================================================
    story.append(Paragraph("6. Presentation Script (3–5 minutes)", S["section"]))

    script_blocks = [
        ("[0:00 – 0:30]  Introduction",
         "Good morning. We are The Prediction XI, and our project uses deep learning to predict "
         "football match outcomes — Home Win, Draw, or Away Win. Since our pitch presentation, "
         "we have completed the entire technical pipeline and are here today to walk you through "
         "our progress, the challenges we faced, and our initial results."),
        ("[0:30 – 1:15]  Data Preparation",
         "We collect match data automatically from football-data.co.uk — no manual download needed. "
         "Our dataset covers five major European leagues across seven seasons: roughly 12,000 matches. "
         "We changed our original data source from Kaggle to this because it integrates directly into "
         "our pipeline with a single command. From the raw data we engineer 35 features per match: "
         "rolling form over the last five games, head-to-head win rates, home and away venue strength, "
         "and relative differentials between the two teams. Crucially, every feature is computed using "
         "only matches before the prediction date — no future data leaks in."),
        ("[1:15 – 2:15]  Models & Results",
         "We trained three models. First, a Random Forest baseline — 300 trees with balanced class "
         "weights — which achieves around 53% accuracy and a Macro F1 of 43%. Second, our main model: "
         "a Deep MLP with three hidden layers, BatchNorm, Dropout, and early stopping, tuned with "
         "Optuna. It reaches approximately 55% accuracy and 46% Macro F1. Third, an LSTM that processes "
         "sequences of five consecutive matches, achieving comparable results to the MLP. "
         "All three models significantly outperform the naive baseline of always predicting a Home Win, "
         "which sits at 46% accuracy but zero F1 on Draws and Away Wins."),
        ("[2:15 – 3:00]  Challenges",
         "Our biggest technical challenge was class imbalance. Draws make up only 26% of matches and "
         "are the hardest outcome to predict for any model. We addressed this with class-weighted loss "
         "functions and see improvement, but Draw prediction remains the weakest class — which is "
         "consistent with what professional betting markets also struggle with. "
         "We also had to be very careful about temporal feature computation: every rolling statistic "
         "must only look backwards in time. A single mistake here would contaminate results with "
         "future information."),
        ("[3:00 – 3:45]  What Remains",
         "We are very close to the finish line. The core technical work is complete and we are "
         "wrapping up the final details. What remains is the poster and the final presentation. "
         "We are also looking at some additional improvements and visual enhancements — better "
         "styled plots, an improved Streamlit interface, and possibly some extra analysis such as "
         "a per-league accuracy breakdown. We also have a working Streamlit web application where "
         "you can select any two teams, pick a date, and instantly see predicted probabilities for "
         "all three outcomes — which we plan to demonstrate live at the final presentation."),
        ("[3:45 – 4:00]  Closing",
         "In summary: the project is in a very strong position. Full pipeline complete, models "
         "trained and evaluated, GitHub repository live, Streamlit app working. We are doing very "
         "well and are excited to present the final result. Thank you — happy to take any questions."),
    ]
    for heading, text in script_blocks:
        story.append(KeepTogether([
            Paragraph(f"<b>{heading}</b>", S["subsection"]),
            Paragraph(text, S["script"]),
            Spacer(1, 0.2*cm),
        ]))

    # ===================================================================
    # SECTION 7 — Next Steps
    # ===================================================================
    story.append(Paragraph("7. Remaining Work", S["section"]))
    next_steps = [
        "Design and print the final A1 poster (Phase 7)",
        "Prepare the final 10-minute presentation slides",
        "Live demo of the Streamlit app during the final presentation",
        "Visual improvements to the project: enhanced plot styling, improved Streamlit UI layout, and additional charts such as a per-league accuracy breakdown and a head-to-head probability comparison view",
        "Optional: add SHAP explainability values to feature importance section",
    ]
    for item in next_steps:
        story.append(Paragraph(f"• {item}", S["bullet"]))

    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width=W, thickness=0.5, color=colors.HexColor("#B0BEC5")))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "The Prediction XI — PAAI Course 2026 | GitHub: The-Prediction-XI",
        S["label"],
    ))

    doc.build(story)
    print(f"PDF saved: {path}")


if __name__ == "__main__":
    build_pdf(OUTPUT)

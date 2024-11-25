import os
import numpy as np
from loguru import logger
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, letter
from reportlab.platypus import Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image


def create_radar_chart(values, categories, filename):
    """
    Generate a radar chart.

    :param values: a list of trait values
    :param categories: a list of trait names
    :param filename: the filename of the generated radar chart image
    :return: None
    """
    N = len(categories)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(3, 3), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color="gray", alpha=0.25)
    ax.plot(angles, values, color="black", linewidth=2)

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)

    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def create_progress_bars(data, body_style):
    """
    Create a table of progress bars.

    :param data: a list of tuples of (trait, value) for each trait
    :param body_style: the style for the table body
    :return: a table of progress bars
    """
    table_data = []
    max_bar_width = 4.5 * inch

    for trait, value in data:
        trait_label = Paragraph(f"<b>{trait}</b> {value}%", body_style)

        bar_width = min(value * 2, max_bar_width)

        bar = Table(
            [[Paragraph(f"<b>{value}%</b>", body_style)]], colWidths=[bar_width]
        )

        bar.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.black),
                    ("TEXTCOLOR", (0, 0), (-1, -1), colors.white),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 2),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ]
            )
        )

        table_data.append([trait_label, bar])

    table = Table(table_data, colWidths=[1.5 * inch, 4.5 * inch])

    table.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
            ]
        )
    )

    return table


def create_proctoring_table(proctoring_data, body_style):
    table_data = [
        [Paragraph(f"{key}", body_style), Paragraph(str(value), body_style)]
        for key, value in proctoring_data.items()
    ]

    table = Table(table_data, colWidths=[3 * inch, 2 * inch])

    table.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
            ]
        )
    )

    return table


def generate_pdf(content, traits, file_name):
    """
    Generate a PDF report with radar chart and progress bars.

    :param content: a list of tuples of (heading, paragraph) for each section of the report
    :param traits: a dictionary of trait names to values
    :param proctoring_data: a dictionary of proctoring data
    :return: None
    """
    logger.info("Pdf generation started")
    pdf_path = os.path.join("reports", file_name)
    pdf = SimpleDocTemplate(pdf_path, pagesize=A4)

    styles = getSampleStyleSheet()
    heading_style = ParagraphStyle(
        "HeadingStyle",
        parent=styles["Heading1"],
        fontSize=10,
        textColor=colors.black,
        spaceAfter=0,
    )
    body_style = ParagraphStyle(
        "BodyStyle", parent=styles["BodyText"], fontSize=7, spaceAfter=0
    )

    categories = [k.capitalize() for k in traits.keys()]
    values = [v for v in traits.values()]

    radar_chart_filename = "radar_chart.png"
    create_radar_chart(values, categories, radar_chart_filename)

    elements = []

    elements.append(Paragraph("Overview", heading_style))

    elements.append(
        Paragraph("The five traits compared to the population average.", body_style)
    )
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Image(radar_chart_filename, width=3.5 * inch, height=3 * inch))
    elements.append(Spacer(1, 0.3 * inch))

    overview_text = content[0][1]
    content.pop(0)
    elements.append(Paragraph(overview_text, body_style))
    elements.append(Spacer(1, 0.3 * inch))

    traits_data = [(k.capitalize(), round(v)) for k, v in traits.items()]
    progress_bars_table = create_progress_bars(traits_data, body_style)
    elements.append(progress_bars_table)
    elements.append(Spacer(1, 0.3 * inch))

    for heading, paragraph_text in content:
        elements.append(Paragraph(heading, heading_style))
        elements.append(Paragraph(paragraph_text, body_style))
        elements.append(Spacer(1, 0.1 * inch))

    pdf.build(elements)

    if os.path.exists(radar_chart_filename):
        os.remove(radar_chart_filename)

    logger.info("PDF generated with radar chart and progress bars!")
    return pdf_path


def generate_proctoring_report(output_file, proctoring_data):
    c = canvas.Canvas(output_file, pagesize=letter)
    width, height = letter

    # Title Section
    c.setFillColor(colors.darkblue)
    c.setFont("Helvetica-Bold", 24)
    c.drawCentredString(width / 2, height - 60, "Proctoring Analysis Report")

    # Horizontal line below the title
    c.setStrokeColor(colors.darkblue)
    c.setLineWidth(2)
    c.line(50, height - 70, width - 50, height - 70)

    y_position = height - 110
    c.setFont("Helvetica-Bold", 14)
    c.drawString(70, y_position, "Summary:")
    # c.setLineWidth(1)
    # c.setStrokeColor(colors.lightgrey)
    # c.line(70, y_position - 5, width - 70, y_position - 5)  # Horizontal line under "Summary"

    image_violations = proctoring_data.get("violated_frames", set())
    images_captured = proctoring_data.get("images_captured", 0)
    summary_data = [
        ("Images Captured", images_captured),
        ("Image Violations", len(image_violations)),
    ]
    c.setFont("Helvetica", 12)
    c.setFillColor(colors.black)
    y_position -= 30
    for label, value in summary_data:
        c.drawString(80, y_position, f"{label}:")
        c.drawRightString(width - 80, y_position, str(value))
        y_position -= 20

    c.setStrokeColor(colors.grey)
    c.line(50, y_position - 10, width - 50, y_position - 10)

    y_position -= 40
    c.setFont("Helvetica-Bold", 14)
    c.setFillColor(colors.darkblue)
    c.drawString(70, y_position, "Detailed Analysis:")
    # c.line(70, y_position - 5, width - 70, y_position - 5)  # Line under "Detailed Analysis"

    details = [
        ("Multiple Face Detection", proctoring_data.get("multiple_faces", 0)),
        ("No Face Detection", proctoring_data.get("no_faces", 0)),
        ("Mobile Phone Detection", proctoring_data.get("mobile_detected", 0)),
        ("Mouth Opening Detection", proctoring_data.get("mouth_open", 0)),
        ("Eye Tracking", proctoring_data.get("eye_tracker", 0)),
        ("Head Pose Detection", proctoring_data.get("head_pose", 0))
    ]
    c.setFont("Helvetica", 12)
    y_position -= 30
    for label, value in details:
        c.setFillColor(colors.black)
        c.drawString(80, y_position, f"{label}:")
        # Use flags to indicate status
        if value == 0:
            c.setFillColor(colors.green)
            c.drawString(width - 150, y_position, "✔ No Violations")
        else:
            c.setFillColor(colors.red)
            c.drawString(width - 150, y_position, "✖ Violations Detected")
        y_position -= 20

    # Footer
    c.setFont("Helvetica-Oblique", 10)
    c.setFillColor(colors.grey)
    c.drawCentredString(width / 2, 50, "Generated by Proctoring System - PrismHire")

    c.save()

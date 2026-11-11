"""
Generate a perfectly balanced 2-Page Progress Report using safe standard encoding.
Page 1: Project Progress, 4 KPI stats, 3 completed milestones, dataset summary table, 3 ML configurations.
Page 2: Exactly the 3 student questions, evaluation metrics, and end-sem timeline.
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas
import fitz

class BalancedNumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_decorations(self, page_count):
        self.saveState()
        # Top banner accent
        self.setFillColor(colors.HexColor("#1D4ED8"))
        self.rect(0, 786, 612, 6, fill=True, stroke=False)
        
        # Header text
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#334155"))
        self.drawString(40, 764, "STUDENT PROJECT PROGRESS REPORT  |  AIML IRRIGATION SUPPORT")
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#047857"))
        self.drawRightString(572, 764, "WEEKS 1 & 2 COMPLETED")
        
        # Header divider
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.8)
        self.line(40, 756, 572, 756)
        
        # Footer divider
        self.line(40, 38, 572, 38)
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        self.drawString(40, 26, "Project 1: Machine Learning-Based ET0 Estimation | NASA POWER Data (Hisar, Haryana)")
        self.setFont("Helvetica-Bold", 8)
        self.drawRightString(572, 26, f"Page {self._pageNumber} of {page_count}")
        self.restoreState()

def build_pdf(filename="Project_Progress_Report_2_Pages.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=46,
        bottomMargin=44
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'MainTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=18,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=2
    )

    meta_style = ParagraphStyle(
        'MainMeta',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#475569"),
        spaceAfter=6
    )

    sec_h1 = ParagraphStyle(
        'SecHeading',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#1D4ED8"),
        spaceBefore=4,
        spaceAfter=3
    )

    body_text = ParagraphStyle(
        'BodyCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#334155"),
        spaceAfter=2
    )

    kpi_val = ParagraphStyle(
        'KPIVal',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=13,
        alignment=1,
        textColor=colors.HexColor("#1E3A8A")
    )

    kpi_lbl = ParagraphStyle(
        'KPILbl',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.2,
        leading=9,
        alignment=1,
        textColor=colors.HexColor("#64748B")
    )

    card_title = ParagraphStyle(
        'CardTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#0F172A")
    )

    card_text = ParagraphStyle(
        'CardText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.8,
        leading=10.5,
        textColor=colors.HexColor("#334155")
    )

    q_title = ParagraphStyle(
        'QTitleStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#1E3A8A"),
        spaceBefore=5,
        spaceAfter=2
    )

    q_box = ParagraphStyle(
        'QBoxStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.3,
        leading=11.8,
        textColor=colors.HexColor("#1E293B"),
        backColor=colors.HexColor("#F8FAFC"),
        borderColor=colors.HexColor("#CBD5E1"),
        borderWidth=0.8,
        borderPadding=6,
        spaceBefore=2,
        spaceAfter=6
    )

    story = []

    # =========================================================================
    # PAGE 1: PROGRESS REPORT
    # =========================================================================
    story.append(Paragraph("Project 1 Progress: Machine Learning-Based ET0 Estimation", title_style))
    story.append(Paragraph("<b>Objective:</b> Estimate daily reference evapotranspiration (ET0) for agricultural irrigation using limited weather variables.", meta_style))

    # KPI Stat Ribbon
    kpi_data = [
        [
            Paragraph("<b>5,113 Days</b>", kpi_val),
            Paragraph("<b>14 Years</b>", kpi_val),
            Paragraph("<b>0 Missing Days</b>", kpi_val),
            Paragraph("<b>100% Match</b>", kpi_val)
        ],
        [
            Paragraph("Daily Observations", kpi_lbl),
            Paragraph("2010 - 2023 Period", kpi_lbl),
            Paragraph("Clean Verified Data", kpi_lbl),
            Paragraph("Exact User CSV Audit", kpi_lbl)
        ]
    ]
    kpi_table = Table(kpi_data, colWidths=[133, 133, 133, 133])
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#EFF6FF")),
        ('BOX', (0,0), (-1,-1), 0.8, colors.HexColor("#BFDBFE")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#DBEAFE")),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 4))

    # Section 1: Completed Milestones
    story.append(Paragraph("1. Work Completed So Far (Weeks 1 & 2 Deliverables)", sec_h1))
    
    milestone_rows = [
        [
            Paragraph("<b>[Step 1 Completed] Meteorological Data Acquisition & Verification (NASA POWER API)</b>", card_title),
        ],
        [
            Paragraph("Fetched 14 years (2010 - 2023) of daily agroclimatology data for <b>Hisar, Haryana</b> (Lat: 29.15 N, Lon: 75.72 E, Elev: 215m) matching the base research paper (Shaloo et al.). Verified 0 missing values, checked physical bounds, and audited against the manual user download with a <b>100.0% row-by-row exact match</b>.", card_text)
        ],
        [
            Paragraph("<b>[Step 2 Completed] Ground-Truth ET0 Formulation (FAO-56 Penman-Monteith)</b>", card_title),
        ],
        [
            Paragraph("Implemented the complete standardized FAO-56 Penman-Monteith physical equation to compute reference evapotranspiration (ET0 in mm/day). The target variable averages <b>5.14 mm/day</b> (0.65 mm in winter to 14.5 mm in peak summer). Verified against Python's certified <code>pyet</code> library with an absolute discrepancy under 0.00012 mm/day.", card_text)
        ],
        [
            Paragraph("<b>[Step 3 Completed] Self-Contained Kaggle / Jupyter Notebook Created</b>", card_title),
        ],
        [
            Paragraph("Created <code>evapotranspiration_fao56_calculation.ipynb</code>. It features automatic data detection for Kaggle input and local folders, handles NASA metadata headers, runs without external dependencies, generates EDA time-series and monthly boxplots, and exports the clean dataset ready for model training.", card_text)
        ]
    ]
    m_table = Table(milestone_rows, colWidths=[532])
    m_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,0), colors.HexColor("#F8FAFC")),
        ('BACKGROUND', (0,2), (0,2), colors.HexColor("#F8FAFC")),
        ('BACKGROUND', (0,4), (0,4), colors.HexColor("#F8FAFC")),
        ('BOX', (0,0), (-1,-1), 0.7, colors.HexColor("#E2E8F0")),
        ('LINEBELOW', (0,1), (-1,1), 0.5, colors.HexColor("#E2E8F0")),
        ('LINEBELOW', (0,3), (-1,3), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(m_table)
    story.append(Spacer(1, 4))

    # Section 2: Dataset Variables Table
    story.append(Paragraph("2. Summary of Weather Variables in Dataset", sec_h1))
    
    var_table_data = [
        ["Variable Name", "Dataset Column", "Unit", "Mean +/- Std", "Range", "Role in Machine Learning"],
        ["Max Temperature", "max_temp", "deg C", "33.15 +/- 7.20", "13.4 - 47.9", "Input Feature (All 3 Configurations)"],
        ["Min Temperature", "min_temp", "deg C", "19.49 +/- 8.18", "0.2 - 34.7", "Input Feature (All 3 Configurations)"],
        ["Relative Humidity", "relative_humidity", "%", "49.20 +/- 18.23", "11.4 - 94.7", "Input Feature (Configurations 2 & 3)"],
        ["Wind Speed (2m)", "wind_speed", "m/s", "2.42 +/- 0.88", "0.6 - 7.9", "Input Feature (Configuration 3 only)"],
        ["Solar Radiation", "solar_radiation", "MJ/m2/d", "17.52 +/- 5.69", "1.4 - 29.9", "Used to calculate ground-truth ET0 target"],
        ["Surface Pressure", "surface_pressure", "kPa", "98.81 +/- 0.54", "96.9 - 100.3", "Atmospheric pressure for psychrometric constant"],
        ["Target ET0", "ET0", "mm/day", "5.14 +/- 2.37", "0.65 - 14.53", "Prediction Target (y) for all ML models"]
    ]
    v_table = Table(var_table_data, colWidths=[82, 88, 42, 72, 60, 188])
    v_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1E3A8A")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 7.2),
        ('GRID', (0,0), (-1,-1), 0.4, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0,0), (-1,-1), 1.8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.8),
        ('ALIGN', (2,0), (4,-1), 'CENTER'),
        ('BACKGROUND', (0,-1), (-1,-1), colors.HexColor("#FEF3C7")),
        ('FONTNAME', (0,-1), (-1,-1), 'Helvetica-Bold'),
    ]))
    story.append(v_table)
    story.append(Spacer(1, 4))

    # Section 3: Feature Configurations
    story.append(Paragraph("3. Next Phase: Machine Learning Configurations to Evaluate", sec_h1))
    
    config_data = [
        [
            Paragraph("<b>Config 1: Temperature Only</b>", card_title),
            Paragraph("<b>Config 2: Temp + Humidity</b>", card_title),
            Paragraph("<b>Config 3: Temp + Humidity + Wind</b>", card_title)
        ],
        [
            Paragraph("Features: <code>max_temp</code>, <code>min_temp</code><br/>Simulates the cheapest agricultural weather station ($5 sensor).", card_text),
            Paragraph("Features: <code>max_temp</code>, <code>min_temp</code>, <code>RH</code><br/>Captures air drying power and vapor pressure deficit.", card_text),
            Paragraph("Features: <code>max_temp</code>, <code>min_temp</code>, <code>RH</code>, <code>WS</code><br/>Includes aerodynamic wind transport of water vapor.", card_text)
        ]
    ]
    cfg_table = Table(config_data, colWidths=[177, 177, 178])
    cfg_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F1F5F9")),
        ('BOX', (0,0), (-1,-1), 0.7, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(cfg_table)
    story.append(Spacer(1, 4))
    
    story.append(Paragraph("<b>Core Research Question:</b> <i>Can machine learning estimate ET0 accurately using only temperature and humidity, allowing farmers to eliminate expensive solar radiation sensors?</i>", body_text))

    # =========================================================================
    # PAGE 2: ONLY THE 3 USER-REQUESTED QUESTIONS & END-SEM ROADMAP
    # =========================================================================
    story.append(PageBreak())

    story.append(Paragraph("Page 2: Student Questions for Project Mentor", title_style))
    story.append(Paragraph("3 practical queries from our student team regarding model training, data splitting, and demonstration requirements.", meta_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1D4ED8"), spaceAfter=8))

    # ONLY THE 3 USER QUESTIONS
    story.append(Paragraph("Question 1: Model Training Guidance", q_title))
    story.append(Paragraph(
        "<b>We haven't learned how to train machine learning models yet in class - how should our team approach training this?</b><br/>"
        "Could you guide us on the starter setup? Is it appropriate for us to use Python's <code>scikit-learn</code> library with standard baseline regressors (Linear Regression, Random Forest Regressor, and SVR) to train on our weather inputs?",
        q_box
    ))

    story.append(Paragraph("Question 2: Dataset Train / Test Split Ratio", q_title))
    story.append(Paragraph(
        "<b>In model training, should our dataset split be 80/20 or 70/20/10?</b><br/>"
        "We have 14 years of daily data (5,113 days). Should we use an <b>80% Train / 20% Test</b> split, or does the course grading rubric require a 3-way <b>70% Train / 20% Validation / 10% Test</b> split?",
        q_box
    ))

    story.append(Paragraph("Question 3: End-Semester Demonstration Timeline", q_title))
    story.append(Paragraph(
        "<b>Can we expect to finish this for a complete working demonstration before end-sem?</b><br/>"
        "What are the specific evaluation deliverables expected from our team during the final project demonstration (e.g. running the live Kaggle notebook, presenting the comparison table, and showing Predicted vs. Actual scatter plots)?",
        q_box
    ))

    story.append(Spacer(1, 6))

    # Target Evaluation Metrics Table
    story.append(Paragraph("Target Evaluation Metrics for Model Comparison", sec_h1))
    metrics_data = [
        [
            Paragraph("<b>MAE (Mean Absolute Error)</b>", card_title),
            Paragraph("<b>RMSE (Root Mean Squared Error)</b>", card_title),
            Paragraph("<b>R2 (Coefficient of Determination)</b>", card_title)
        ],
        [
            Paragraph("Measures average magnitude of prediction error in mm/day. Target benchmark: &lt; 0.6 mm/day.", card_text),
            Paragraph("Penalizes large errors heavily; important during heatwave spikes. Target: &lt; 0.9 mm/day.", card_text),
            Paragraph("Percentage of daily ET0 variance explained by the model. Target benchmark: &gt; 0.90.", card_text)
        ]
    ]
    met_table = Table(metrics_data, colWidths=[177, 177, 178])
    met_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8FAFC")),
        ('BOX', (0,0), (-1,-1), 0.7, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(met_table)
    story.append(Spacer(1, 6))

    # Complete Demonstration Plan Before End-Sem
    story.append(Paragraph("Roadmap to Complete Demonstration Before End-Sem", sec_h1))
    
    plan_data = [
        [Paragraph("<b>Step 1: Train the 3 Models (Week 3):</b> Implement Linear Regression, Random Forest, and SVR across the 3 weather configurations (9 total runs).", card_text)],
        [Paragraph("<b>Step 2: Build Evaluation Table (Week 3):</b> Compile test scores for MAE, RMSE, and R2 into a side-by-side comparison matrix.", card_text)],
        [Paragraph("<b>Step 3: Generate Visual Plots (Week 4):</b> Produce Predicted vs. Actual scatter plots and Random Forest feature importance rankings.", card_text)],
        [Paragraph("<b>Step 4: Final Live Demo & Viva (Before End-Sem):</b> Demonstrate the complete Kaggle notebook and answer the core research question.", card_text)]
    ]
    p_table = Table(plan_data, colWidths=[532])
    p_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F0FDF4")),
        ('BOX', (0,0), (-1,-1), 0.7, colors.HexColor("#86EFAC")),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(p_table)

    doc.build(story, canvasmaker=BalancedNumberedCanvas)
    print(f"ReportLab PDF built successfully: {filename}")

if __name__ == "__main__":
    build_pdf()

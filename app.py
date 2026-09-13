import streamlit as st
from PIL import Image
from ultralytics import YOLO
from safety_rules import analyze_safety
import tempfile
import os
from datetime import datetime


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="AI Safety Inspector",
    page_icon="🦺",
    layout="wide"
)


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "models/best.pt"


# ============================================================
# Load Trained YOLO Model
# ============================================================

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


model = load_model()


# ============================================================
# Application Header
# ============================================================

st.title("🦺 AI PPE & Workplace Safety Inspector")

st.write(
    "AI-powered construction safety inspection using "
    "a custom-trained YOLO11 object detection model."
)

st.divider()


# ============================================================
# Upload Workplace Image
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a workplace image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# Main Application
# ============================================================

if uploaded_file is not None:

    # --------------------------------------------------------
    # Load uploaded image
    # --------------------------------------------------------

    image = Image.open(uploaded_file)

    # --------------------------------------------------------
    # Save image temporarily
    # --------------------------------------------------------

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".jpg"
    ) as temp_file:

        image.save(temp_file.name)
        temp_path = temp_file.name


    try:

        # ====================================================
        # YOLO Detection
        # ====================================================

        results = model(
            temp_path,
            conf=0.30
        )

        result = results[0]


        # ====================================================
        # Extract Detection Results
        # ====================================================

        detections = []

        if result.boxes is not None:

            for box in result.boxes:

                class_id = int(box.cls[0])

                confidence = float(
                    box.conf[0]
                )

                class_name = result.names[
                    class_id
                ]

                # Bounding box coordinates
                x1, y1, x2, y2 = (
                    box.xyxy[0].tolist()
                )

                detections.append(
                    {
                        "class": class_name,
                        "confidence": confidence,
                        "bbox": [
                            x1,
                            y1,
                            x2,
                            y2
                        ]
                    }
                )


        # ====================================================
        # Safety Rule Engine
        # ====================================================

        report = analyze_safety(
            detections
        )


        # ====================================================
        # Inspection Summary
        # ====================================================

        st.subheader(
            "📊 Inspection Summary"
        )

        col1, col2, col3, col4 = (
            st.columns(4)
        )


        # ----------------------------------------------------
        # Workers
        # ----------------------------------------------------

        with col1:

            st.metric(
                "Workers",
                report["workers"]
            )


        # ----------------------------------------------------
        # Helmet Compliant
        # ----------------------------------------------------

        with col2:

            helmet_count = sum(
                worker["helmet"]
                for worker
                in report["worker_results"]
            )

            st.metric(
                "Helmet Compliant",
                helmet_count
            )


        # ----------------------------------------------------
        # Helmet Compliance
        # ----------------------------------------------------

        with col3:

            st.metric(
                "Helmet Compliance",
                f'{report["helmet_compliance"]}%'
            )


        # ----------------------------------------------------
        # Vest Compliance
        # ----------------------------------------------------

        with col4:

            st.metric(
                "Vest Compliance",
                f'{report["vest_compliance"]}%'
            )


        st.divider()


        # ====================================================
        # Original vs AI Detection
        # ====================================================

        col1, col2 = st.columns(2)


        # ----------------------------------------------------
        # Original Image
        # ----------------------------------------------------

        with col1:

            st.subheader(
                "Original Image"
            )

            st.image(
                image,
                width="stretch"
            )


        # ----------------------------------------------------
        # AI Detection
        # ----------------------------------------------------

        with col2:

            st.subheader(
                "AI Detection"
            )

            annotated_image = result.plot()

            st.image(
                annotated_image,
                channels="BGR",
                width="stretch"
            )


        st.divider()


        # ====================================================
        # Worker-Level Safety Analysis
        # ====================================================

        st.subheader(
            "👷 Worker-Level Safety Analysis"
        )


        if report["worker_results"]:

            for worker in report[
                "worker_results"
            ]:

                helmet = worker[
                    "helmet"
                ]

                vest = worker[
                    "vest"
                ]

                safe = (
                    helmet and vest
                )


                # ------------------------------------------------
                # Worker status
                # ------------------------------------------------

                if safe:

                    status = "✅ SAFE"

                else:

                    status = "⚠️ VIOLATION"


                st.markdown(
                    f"### Worker "
                    f"{worker['worker_id']} — "
                    f"{status}"
                )


                col1, col2, col3 = (
                    st.columns(3)
                )


                # ------------------------------------------------
                # Helmet status
                # ------------------------------------------------

                with col1:

                    if helmet:

                        st.success(
                            f"⛑️ Helmet: YES "
                            f"({worker['helmet_confidence']:.2f})"
                        )

                    else:

                        st.error(
                            "⛑️ Helmet: "
                            "NOT DETECTED"
                        )


                # ------------------------------------------------
                # Vest status
                # ------------------------------------------------

                with col2:

                    if vest:

                        st.success(
                            f"🦺 Vest: YES "
                            f"({worker['vest_confidence']:.2f})"
                        )

                    else:

                        st.error(
                            "🦺 Vest: "
                            "NOT DETECTED"
                        )


                # ------------------------------------------------
                # Overall worker status
                # ------------------------------------------------

                with col3:

                    if safe:

                        st.success(
                            "STATUS: "
                            "PPE COMPLIANT"
                        )

                    else:

                        st.warning(
                            "STATUS: "
                            "ACTION REQUIRED"
                        )


                st.divider()


        else:

            st.info(
                "No workers detected in "
                "this image."
            )


        # ====================================================
        # Safety Alerts
        # ====================================================

        st.subheader(
            "⚠️ Safety Alerts"
        )


        if report["violations"]:

            for violation in report[
                "violations"
            ]:

                st.warning(
                    violation
                )

        else:

            st.success(
                "No PPE violations detected."
            )


        st.divider()


        # ====================================================
        # Inspection Report
        # ====================================================

        st.subheader(
            "📄 Inspection Report"
        )


        inspection_time = (
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )


        report_lines = []


        # ----------------------------------------------------
        # Report Header
        # ----------------------------------------------------

        report_lines.append(
            "AI PPE & WORKPLACE SAFETY "
            "INSPECTION REPORT"
        )

        report_lines.append(
            "=" * 55
        )

        report_lines.append(
            f"Inspection Time: "
            f"{inspection_time}"
        )

        report_lines.append("")


        # ----------------------------------------------------
        # Overall Results
        # ----------------------------------------------------

        report_lines.append(
            f"Workers Detected: "
            f"{report['workers']}"
        )

        report_lines.append(
            f"Helmet Compliance: "
            f"{report['helmet_compliance']}%"
        )

        report_lines.append(
            f"Vest Compliance: "
            f"{report['vest_compliance']}%"
        )

        report_lines.append("")


        # ----------------------------------------------------
        # Worker Analysis
        # ----------------------------------------------------

        report_lines.append(
            "WORKER ANALYSIS"
        )

        report_lines.append(
            "-" * 55
        )


        for worker in report[
            "worker_results"
        ]:

            helmet_status = (
                "YES"
                if worker["helmet"]
                else "NO"
            )

            vest_status = (
                "YES"
                if worker["vest"]
                else "NO"
            )

            status = (
                "COMPLIANT"
                if (
                    worker["helmet"]
                    and worker["vest"]
                )
                else "VIOLATION"
            )


            report_lines.append(
                f"Worker "
                f"{worker['worker_id']}"
            )

            report_lines.append(
                f"  Helmet: "
                f"{helmet_status}"
            )

            report_lines.append(
                f"  Vest: "
                f"{vest_status}"
            )

            report_lines.append(
                f"  Status: "
                f"{status}"
            )

            report_lines.append("")


        # ----------------------------------------------------
        # Safety Alerts
        # ----------------------------------------------------

        report_lines.append(
            "SAFETY ALERTS"
        )

        report_lines.append(
            "-" * 55
        )


        if report["violations"]:

            for violation in report[
                "violations"
            ]:

                report_lines.append(
                    f"- {violation}"
                )

        else:

            report_lines.append(
                "No violations detected."
            )


        # ----------------------------------------------------
        # Create report text
        # ----------------------------------------------------

        report_text = "\n".join(
            report_lines
        )


        # ----------------------------------------------------
        # Download button
        # ----------------------------------------------------

        st.download_button(
            label=(
                "📥 Download "
                "Inspection Report"
            ),
            data=report_text,
            file_name=(
                "safety_inspection_report.txt"
            ),
            mime="text/plain"
        )


        # ====================================================
        # Detection Details
        # ====================================================

        with st.expander(
            "🔍 Detection Details"
        ):

            if detections:

                for detection in detections:

                    st.write(
                        f"**{detection['class']}** "
                        f"— confidence: "
                        f"{detection['confidence']:.2f}"
                    )

            else:

                st.write(
                    "No objects detected."
                )


    finally:

        # ====================================================
        # Cleanup temporary file
        # ====================================================

        if os.path.exists(temp_path):

            os.unlink(temp_path)
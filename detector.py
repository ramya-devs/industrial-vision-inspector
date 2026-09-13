from ultralytics import YOLO
from safety_rules import analyze_safety


MODEL_PATH = "models/best.pt"

model = YOLO(MODEL_PATH)


def detect_image(image_path, confidence=0.30):

    results = model(
        image_path,
        conf=confidence
    )

    detections = []

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            class_id = int(box.cls[0])
            conf = float(box.conf[0])

            class_name = result.names[class_id]

            # Bounding box coordinates
            x1, y1, x2, y2 = (
                box.xyxy[0].tolist()
            )

            detections.append({
                "class": class_name,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2]
            })

    safety_report = analyze_safety(
        detections
    )

    return results, detections, safety_report


if __name__ == "__main__":

    results, detections, report = detect_image(
        "data/test.jpg"
    )

    print("\n========== SAFETY REPORT ==========\n")

    print(
        f"Workers detected: "
        f"{report['workers']}"
    )

    print(
        f"Helmet compliance: "
        f"{report['helmet_compliance']}%"
    )

    print(
        f"Vest compliance: "
        f"{report['vest_compliance']}%"
    )

    print("\n========== WORKER ANALYSIS ==========\n")

    for worker in report["worker_results"]:

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

        print(
            f"Worker {worker['worker_id']}: "
            f"Helmet={helmet_status}, "
            f"Vest={vest_status}"
        )

    print("\n========== VIOLATIONS ==========\n")

    if report["violations"]:

        for violation in report["violations"]:
            print(f"WARNING: {violation}")

    else:

        print("No violations detected.")

    print("\n=======================================\n")

    for result in results:
        result.show()
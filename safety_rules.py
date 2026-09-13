def point_inside_box(point, box):
    """Check whether a point is inside a bounding box."""
    x, y = point
    x1, y1, x2, y2 = box

    return x1 <= x <= x2 and y1 <= y <= y2


def box_center(box):
    """Return the center point of a bounding box."""
    x1, y1, x2, y2 = box

    return (
        (x1 + x2) / 2,
        (y1 + y2) / 2
    )


def associate_ppe_to_workers(detections):
    """
    Associate helmets and vests with detected workers
    using bounding-box positions.
    """

    workers = [
        d for d in detections
        if d["class"] == "Person"
    ]

    helmets = [
        d for d in detections
        if d["class"] == "helmet"
    ]

    vests = [
        d for d in detections
        if d["class"] == "vest"
    ]

    worker_results = []

    for index, worker in enumerate(workers, start=1):

        worker_box = worker["bbox"]

        x1, y1, x2, y2 = worker_box

        worker_height = y2 - y1

        # --------------------------------------------------
        # Find helmet belonging to this worker
        # --------------------------------------------------

        assigned_helmet = None

        for helmet in helmets:

            center = box_center(helmet["bbox"])

            # Helmet should be inside/near upper part of person
            if point_inside_box(center, worker_box):

                helmet_y = center[1]

                upper_limit = y1 + (worker_height * 0.45)

                if helmet_y <= upper_limit:

                    if (
                        assigned_helmet is None
                        or helmet["confidence"]
                        > assigned_helmet["confidence"]
                    ):
                        assigned_helmet = helmet

        # --------------------------------------------------
        # Find vest belonging to this worker
        # --------------------------------------------------

        assigned_vest = None

        for vest in vests:

            center = box_center(vest["bbox"])

            if point_inside_box(center, worker_box):

                vest_y = center[1]

                # Vest should be around torso area
                lower_limit = y1 + (worker_height * 0.75)

                upper_limit = y1 + (worker_height * 0.15)

                if upper_limit <= vest_y <= lower_limit:

                    if (
                        assigned_vest is None
                        or vest["confidence"]
                        > assigned_vest["confidence"]
                    ):
                        assigned_vest = vest

        has_helmet = assigned_helmet is not None
        has_vest = assigned_vest is not None

        worker_results.append({
            "worker_id": index,
            "helmet": has_helmet,
            "vest": has_vest,
            "helmet_confidence": (
                assigned_helmet["confidence"]
                if assigned_helmet
                else 0
            ),
            "vest_confidence": (
                assigned_vest["confidence"]
                if assigned_vest
                else 0
            ),
        })

    return worker_results


def analyze_safety(detections):

    worker_results = associate_ppe_to_workers(
        detections
    )

    workers = len(worker_results)

    helmet_compliant = sum(
        worker["helmet"]
        for worker in worker_results
    )

    vest_compliant = sum(
        worker["vest"]
        for worker in worker_results
    )

    violations = []

    for worker in worker_results:

        if not worker["helmet"]:

            violations.append(
                f"Worker {worker['worker_id']}: "
                "No helmet detected."
            )

        if not worker["vest"]:

            violations.append(
                f"Worker {worker['worker_id']}: "
                "No vest detected."
            )

    helmet_compliance = (
        helmet_compliant / workers * 100
        if workers > 0
        else 0
    )

    vest_compliance = (
        vest_compliant / workers * 100
        if workers > 0
        else 0
    )

    return {
        "workers": workers,
        "helmet_compliance": round(
            helmet_compliance, 1
        ),
        "vest_compliance": round(
            vest_compliance, 1
        ),
        "worker_results": worker_results,
        "violations": violations,
    }
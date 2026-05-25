import math

# Thresholds for normalization based on dataset properties
SAT_CTR_REL = 0.20
SAT_CENTAR = 0.15
SAT_OSI = 0.30
SAT_KUT = 30.0 # degrees

# Weights for the combined error function
W_CTR = 0.40
W_IOU = 0.25
W_CENTAR = 0.15
W_OSI = 0.10
W_KUT = 0.10

def calculate_centroid_error(auto, gt):
    """Distance between centers normalized by the mean radius of GT ellipse."""
    if auto is None or gt is None:
        return None
    d = math.hypot(auto["cx"] - gt["cx"], auto["cy"] - gt["cy"])
    r_ref = (gt["a"] + gt["b"]) / 2.0
    return d / r_ref if r_ref > 0 else None

def calculate_axis_error(auto, gt):
    """Relative error of semi-axes a and b."""
    if auto is None or gt is None or gt["a"] <= 0 or gt["b"] <= 0:
        return None
    ea = abs(auto["a"] - gt["a"]) / gt["a"]
    eb = abs(auto["b"] - gt["b"]) / gt["b"]
    return (ea + eb) / 2.0

def calculate_angle_error(auto, gt):
    """Minimum angle difference taking 180-degree symmetry into account."""
    if auto is None or gt is None:
        return None
    diff = abs(auto["kut"] - gt["kut"]) % 180
    return min(diff, 180 - diff)

def _bounded(x, threshold):
    """Bounds error metric to a max of 1.0 based on threshold."""
    if x is None or threshold <= 0:
        return None
    return min(1.0, x / threshold)

def calculate_combined_error(m):
    """
    Combined error E normalized to 0-1 range.
    Lower is better; CTR has highest weight as it's the main clinical measure.
    The dictionary 'm' must contain:
    ctr_err_rel, cent_s, cent_t, osi_s, osi_t, iou_s, iou_t, kut_s, kut_t
    """
    required = [
        "ctr_err_rel",
        "cent_s", "cent_t",
        "osi_s", "osi_t",
        "iou_s", "iou_t",
        "kut_s", "kut_t"
    ]
    if any(m.get(k) is None for k in required):
        return None, None

    e_ctr = _bounded(m["ctr_err_rel"], SAT_CTR_REL)
    e_centar = _bounded((m["cent_s"] + m["cent_t"]) / 2.0, SAT_CENTAR)
    e_osi = _bounded((m["osi_s"] + m["osi_t"]) / 2.0, SAT_OSI)
    e_iou = 1.0 - (m["iou_s"] + m["iou_t"]) / 2.0
    e_kut = _bounded((m["kut_s"] + m["kut_t"]) / 2.0, SAT_KUT)

    # Note: e_iou is already between 0 and 1, so no bounded needed.
    E = W_CTR * e_ctr + W_IOU * e_iou + W_CENTAR * e_centar + W_OSI * e_osi + W_KUT * e_kut
    
    if E <= 0.20:
        category = "excellent"
    elif E <= 0.40:
        category = "good"
    elif E <= 0.60:
        category = "poor"
    else:
        category = "bad"
        
    return float(E), category

def calculate_all_metrics(auto_cardiac, auto_thorax, gt_cardiac, gt_thorax, iou_srce, iou_toraks):
    """
    A helper function to calculate all the metrics at once.
    """
    # Calculate CTR errors
    ctr_auto = auto_cardiac["a"] / auto_thorax["a"] if auto_cardiac and auto_thorax and auto_thorax["a"] > 0 else None
    ctr_gt = gt_cardiac["a"] / gt_thorax["a"] if gt_cardiac and gt_thorax and gt_thorax["a"] > 0 else None
    
    abs_err = abs(ctr_auto - ctr_gt) if (ctr_auto is not None and ctr_gt is not None) else None
    rel_err = abs_err / ctr_gt if (abs_err is not None and ctr_gt is not None and ctr_gt > 0) else None

    # Gather data for combined error
    m = {
        "ctr_err_rel": rel_err,
        "cent_s": calculate_centroid_error(auto_cardiac, gt_cardiac),
        "cent_t": calculate_centroid_error(auto_thorax, gt_thorax),
        "osi_s": calculate_axis_error(auto_cardiac, gt_cardiac),
        "osi_t": calculate_axis_error(auto_thorax, gt_thorax),
        "iou_s": iou_srce,
        "iou_t": iou_toraks,
        "kut_s": calculate_angle_error(auto_cardiac, gt_cardiac),
        "kut_t": calculate_angle_error(auto_thorax, gt_thorax)
    }

    combined_e, category = calculate_combined_error(m)
    
    return {
        "ctr_auto": ctr_auto,
        "ctr_gt": ctr_gt,
        "abs_err": abs_err,
        "rel_err": rel_err,
        "combined_err": combined_e,
        "category": category,
        "metrics_dict": m
    }

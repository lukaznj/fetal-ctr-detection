import csv
import json
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse

def visualize(img_bgr, gray, edges, detection, annotation,
              metrics_res, name, output_dir: Path):
    """
    Step 7: Plotting detected and GT ellipses on the original image.
    """
    ctr_auto = metrics_res['ctr_auto']
    ctr_gt = metrics_res['ctr_gt']
    ctr_auto_str = f"{ctr_auto:.3f}" if ctr_auto is not None else "N/A"
    ctr_gt_str = f"{ctr_gt:.3f}" if ctr_gt is not None else "N/A"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Image {name} | CTR auto: {ctr_auto_str} | CTR GT: {ctr_gt_str}",
        fontsize=12
    )

    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title("Original Grayscale")
    axes[0].axis("off")

    axes[1].imshow(edges, cmap="gray")
    axes[1].set_title("Edge Detection")
    axes[1].axis("off")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    axes[2].imshow(img_rgb)
    axes[2].set_title("Auto ellipses & GT annotations")
    axes[2].axis("off")

    ax = axes[2]

    if detection:
        for cls, ellipse in detection.items():
            if ellipse is None:
                continue
            e = Ellipse(
                xy=(ellipse["cx"], ellipse["cy"]),
                width=2 * ellipse["a"],
                height=2 * ellipse["b"],
                angle=ellipse["kut"],
                edgecolor="red",
                facecolor="none",
                linewidth=2,
                linestyle="--"
            )
            ax.add_patch(e)
            ax.text(
                ellipse["cx"], ellipse["cy"] - ellipse["b"] - 8,
                f"auto {cls}",
                color="red", fontsize=8, ha="center"
            )

    if annotation:
        for cls, ann in annotation.items():
            e = Ellipse(
                xy=(ann["cx"], ann["cy"]),
                width=2 * ann["a"],
                height=2 * ann["b"],
                angle=ann["kut"],
                edgecolor="lime",
                facecolor="none",
                linewidth=2
            )
            ax.add_patch(e)
            ax.text(
                ann["cx"], ann["cy"] + ann["b"] + 12,
                f"GT {cls}",
                color="lime", fontsize=8, ha="center"
            )

    red_patch = mpatches.Patch(color="red", label="Automatic detection")
    green_patch = mpatches.Patch(color="lime", label="Ground truth")
    ax.legend(handles=[red_patch, green_patch], loc="lower right", fontsize=8)

    plt.tight_layout()
    path = output_dir / f"{name}_result.png"
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close()

def save_graphs(errors, ctr_auto, ctr_gt, output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Accuracy Analysis of CTR Detection", fontsize=14)

    axes[0].hist(errors, bins=15, color="steelblue", edgecolor="white", alpha=0.85)
    axes[0].axvline(np.mean(errors), color="red", linestyle="--",
                    label=f"MAE = {np.mean(errors):.3f}")
    axes[0].set_xlabel("Absolute CTR Error")
    axes[0].set_ylabel("Number of images")
    axes[0].set_title("Error Distribution")
    axes[0].legend()

    if ctr_auto and ctr_gt:
        n = min(len(ctr_auto), len(ctr_gt))
        axes[1].scatter(ctr_gt[:n], ctr_auto[:n], alpha=0.65,
                        color="steelblue", edgecolors="navy")
        maximum = max(max(ctr_gt[:n]), max(ctr_auto[:n])) * 1.1
        lim = [0, maximum]
        axes[1].plot(lim, lim, "r--", label="Ideal match")
        axes[1].set_xlabel("CTR Ground Truth")
        axes[1].set_ylabel("CTR Automatic Detection")
        axes[1].set_title("Auto vs GT CTR")
        axes[1].legend()
        axes[1].set_xlim(lim)
        axes[1].set_ylim(lim)

    plt.tight_layout()
    path = output_dir / "accuracy_analysis.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Accuracy graph saved: {path}")


def save_results(results: list[dict], output_dir: Path):
    path = output_dir / "results_per_image.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        fields = ["name", "ctr_auto", "ctr_gt", "abs_err", "rel_err",
                  "combined_err", "iou_heart", "iou_thorax", "n_candidates"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        
        for r in results:
            m = r["metrics"]
            writer.writerow({
                "name": r["name"],
                "ctr_auto": m["ctr_auto"],
                "ctr_gt": m["ctr_gt"],
                "abs_err": m["abs_err"],
                "rel_err": m["rel_err"],
                "combined_err": m["combined_err"],
                "iou_heart": m["metrics_dict"]["iou_s"],
                "iou_thorax": m["metrics_dict"]["iou_t"],
                "n_candidates": r["n_candidates"]
            })
    print(f"Results per image saved: {path}")

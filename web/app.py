import time
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

import gradio as gr
import group

MODEL_CONFIGS: Dict[str, Dict] = {
    "LEVIR-CD · Dual-Stage Change Detection": {
        "s1_weights": r"levir_s1.pth",
        "s2_weights": r"levir_s2.pth",
        "tile_size": 512,
        "dataset": "LEVIR-CD",
        "backbone": "ResNet-50",
        "type": "Dual-Stage CD Network",
        "task": "Image Change Detection",
    },
    "LIM-CD · Lightweight Image Matching CD": {
        "s1_weights": r"lim_s1.pth",
        "s2_weights": r"lim_s2.pth",
        "tile_size": 512,
        "dataset": "LIM-CD",
        "backbone": "ResNet-50",
        "type": "Dual-Stage CD Network",
        "task": "Image Change Detection",
    },
}

DEFAULT_MODEL_NAME = list(MODEL_CONFIGS.keys())[0]
TILESIZE = MODEL_CONFIGS[DEFAULT_MODEL_NAME]["tile_size"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if not hasattr(group.ChangeDetectionNet, "forward_with_stage2_features") and hasattr(group, "_forward_s2"):
    group.ChangeDetectionNet.forward_with_stage2_features = group._forward_s2

if hasattr(group, "S2_CONFIG") and hasattr(group, "DATASET"):
    S2_CFG = group.S2_CONFIG[group.DATASET]
else:
    S2_CFG = {"aspp_dilations": (1, 3, 6), "delta_scale_init": 0.5}

_MODEL_CACHE: Dict[str, Tuple[group.ChangeDetectionNet, group.ChangeRefinementNet]] = {}
AUTO_THRESHOLDS = np.round(np.arange(0.1, 0.91, 0.1), 2)


def _build_model_pair(model_name: str):
    cfg = MODEL_CONFIGS[model_name]
    s1 = group.ChangeDetectionNet().to(DEVICE)
    s1.load_state_dict(torch.load(cfg["s1_weights"], map_location=DEVICE))
    s1.eval()
    for p in s1.parameters():
        p.requires_grad = False

    s2 = group.ChangeRefinementNet(
        aspp_dilations=S2_CFG["aspp_dilations"],
        delta_scale_init=S2_CFG["delta_scale_init"],
    ).to(DEVICE)
    s2.load_state_dict(torch.load(cfg["s2_weights"], map_location=DEVICE))
    s2.eval()
    for p in s2.parameters():
        p.requires_grad = False
    return s1, s2


def get_model_pair(model_name: str):
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = _build_model_pair(model_name)
    return _MODEL_CACHE[model_name]


_UI_MEAN = [0.485, 0.456, 0.406]
_UI_STD = [0.229, 0.224, 0.225]
_pre_tf = transforms.Compose([
    transforms.Resize((TILESIZE, TILESIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_UI_MEAN, std=_UI_STD),
])


def _to_tensor(img: Image.Image) -> torch.Tensor:
    if img.mode != "RGB":
        img = img.convert("RGB")
    return _pre_tf(img).unsqueeze(0).to(DEVICE)


def _denorm(batch: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(_UI_MEAN, device=batch.device).view(1, 3, 1, 1)
    std = torch.tensor(_UI_STD, device=batch.device).view(1, 3, 1, 1)
    img = (batch * std + mean).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    return (img * 255).astype(np.uint8)


def _overlay(base_rgb: np.ndarray, mask01: np.ndarray, color=(255, 120, 0), alpha: float = 0.55) -> np.ndarray:
    base = base_rgb.astype(np.float32).copy()
    c = np.array(color, dtype=np.float32)
    m = mask01.astype(bool)
    base[m] = base[m] * (1.0 - alpha) + c * alpha
    return base.clip(0, 255).astype(np.uint8)


def _mask_from_pil(mask_img: Image.Image) -> np.ndarray:
    m = np.array(mask_img.convert("L"))
    m = cv2.resize(m, (TILESIZE, TILESIZE), interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.uint8)


def _bbox_count(mask01: np.ndarray) -> int:
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask01.astype(np.uint8), connectivity=8)
    count = 0
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 16:
            count += 1
    return count


def _draw_bboxes(base_rgb: np.ndarray, mask01: np.ndarray) -> np.ndarray:
    canvas = base_rgb.copy()
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask01.astype(np.uint8), connectivity=8)
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 16:
            continue
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 255), 2)
    return canvas


def _mask_to_rgb(mask01: np.ndarray) -> np.ndarray:
    return np.stack([mask01 * 255] * 3, axis=-1).astype(np.uint8)


def _draw_contours(base_rgb: np.ndarray, mask01: np.ndarray) -> np.ndarray:
    canvas = base_rgb.copy()
    contours, _ = cv2.findContours(mask01.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, (0, 255, 120), 2)
    return canvas


def _compare_t1_t2(t1_rgb: np.ndarray, t2_rgb: np.ndarray) -> np.ndarray:
    return np.concatenate([t1_rgb, t2_rgb], axis=1)


def _compare_before_after_overlay(t2_rgb: np.ndarray, overlay_rgb: np.ndarray) -> np.ndarray:
    return np.concatenate([t2_rgb, overlay_rgb], axis=1)


def _confidence_focus(prob: np.ndarray, threshold: float) -> np.ndarray:
    dist = np.abs(prob - threshold)
    score = 1.0 - np.clip(dist / 0.5, 0, 1)
    img = (score * 255).astype(np.uint8)
    return cv2.applyColorMap(img, cv2.COLORMAP_INFERNO)[:, :, ::-1]


def _model_info(model_name: str):
    cfg = MODEL_CONFIGS[model_name]
    return {
        "Model Type": cfg["type"],
        "Backbone": cfg["backbone"],
        "Task": cfg["task"],
        "Dataset": cfg["dataset"],
    }


def _safe_metrics(pred01: np.ndarray, gt01):
    if gt01 is None:
        return {"accuracy": None, "precision": None, "recall": None, "iou": None, "f1": None}
    return group.compute_change_metrics(pred01, gt01)


def _choose_best_threshold(prob: np.ndarray, gt01: np.ndarray = None):
    best_threshold = 0.5
    best_score = -1.0
    best_metric_name = "Confidence proxy"
    for thr in AUTO_THRESHOLDS:
        pred01 = (prob > float(thr)).astype(np.uint8)
        changed_px = int(pred01.sum())
        change_pct = 100.0 * changed_px / max(1, pred01.size)
        conf_pct = float(prob[pred01 == 1].mean() * 100.0) if changed_px > 0 else 0.0
        if gt01 is not None:
            metrics = group.compute_change_metrics(pred01, gt01)
            score = float(metrics["f1"])
            if score > best_score:
                best_score = score
                best_threshold = float(thr)
                best_metric_name = "F1"
        else:
            score = conf_pct - 0.15 * abs(change_pct - 5.0)
            if score > best_score:
                best_score = score
                best_threshold = float(thr)
                best_metric_name = "Confidence proxy"
    return best_threshold, best_metric_name


def run_dashboard(img_t1, img_t2, gt_mask, model_name, threshold):
    if img_t1 is None or img_t2 is None:
        raise gr.Error("Please upload both T1 and T2 images.")

    # Fast path: if the two inputs are exactly the same image after resize/convert,
    # short-circuit to an all-zero change mask so every model behaves consistently.
    base1 = np.array(img_t1.resize((TILESIZE, TILESIZE)).convert("RGB"))
    base2 = np.array(img_t2.resize((TILESIZE, TILESIZE)).convert("RGB"))
    identical_inputs = np.array_equal(base1, base2)

    started = time.time()
    s1_infer, s2_infer = get_model_pair(model_name)

    with torch.no_grad():
        t1 = _to_tensor(Image.fromarray(base1))
        t2 = _to_tensor(Image.fromarray(base2))
        if identical_inputs:
            # Skip network pass entirely; probability map is all zeros.
            prob = np.zeros((TILESIZE, TILESIZE), dtype=np.float32)
        else:
            use_amp = DEVICE == "cuda"
            with torch.amp.autocast("cuda", enabled=use_amp):
                p1_logit, diff4, diff3_attn, diff2_raw, _ = s1_infer.forward_with_stage2_features(t1, t2)
                p2_logit, _, _ = s2_infer(diff3_attn, diff4, diff2_raw, p1_logit)
            prob = torch.sigmoid(p2_logit)[0, 0].cpu().numpy()

    t1_vis = _denorm(t1)
    t2_vis = _denorm(t2)
    gt01 = _mask_from_pil(gt_mask) if gt_mask is not None else None

    best_threshold, best_metric_name = _choose_best_threshold(prob, gt01)
    selected_threshold = float(threshold)

    pred_best = (prob > best_threshold).astype(np.uint8)
    pred_selected = (prob > selected_threshold).astype(np.uint8)

    elapsed = time.time() - started

    # Main displayed outputs follow selected threshold; best threshold is only for metrics row.
    prob_vis = cv2.applyColorMap((prob * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[:, :, ::-1]
    overlay_selected = _overlay(t2_vis, pred_selected, color=(255, 140, 0), alpha=0.55)
    bbox_image = _draw_bboxes(t2_vis, pred_selected)
    binary_mask = _mask_to_rgb(pred_selected)
    contour_image = _draw_contours(t2_vis, pred_selected)
    before_after_image = _compare_t1_t2(t1_vis, t2_vis)
    compare_overlay_image = _compare_before_after_overlay(t2_vis, overlay_selected)
    confidence_focus_image = _confidence_focus(prob, selected_threshold)

    changed_px = int(pred_selected.sum())
    total_px = pred_selected.size
    change_pct = 100.0 * changed_px / max(1, total_px)
    conf_pct = float(prob[pred_selected == 1].mean() * 100.0) if changed_px > 0 else 0.0
    bbox_count = _bbox_count(pred_selected)

    best_metrics = _safe_metrics(pred_best, gt01)
    selected_metrics = _safe_metrics(pred_selected, gt01)

    return {
        "best_threshold": best_threshold,
        "selected_threshold": selected_threshold,
        "best_metric_name": best_metric_name,
        "bbox_image": bbox_image,
        "binary_mask": binary_mask,
        "prob_vis": prob_vis,
        "overlay_pred": overlay_selected,
        "contour_image": contour_image,
        "before_after_image": before_after_image,
        "compare_overlay_image": compare_overlay_image,
        "confidence_focus_image": confidence_focus_image,
        "change_pct": f"{change_pct:.2f}%",
        "conf_pct": f"{conf_pct:.2f}%",
        "regions": str(bbox_count),
        "infer_time": f"{elapsed:.3f}s",
        "best_metrics": best_metrics,
        "selected_metrics": selected_metrics,
    }



CSS = r"""
html, body {
    margin: 0;
    min-height: 100%;
    background: radial-gradient(circle at 20% 15%, rgba(22,55,150,0.18), transparent 26%), radial-gradient(circle at 80% 80%, rgba(68,0,125,0.12), transparent 24%), linear-gradient(180deg, #050816 0%, #02040d 100%);
}
.gradio-container {
    max-width: 1450px !important;
    margin: 0 auto;
    padding: 14px !important;
    font-family: Inter, system-ui, sans-serif;
}
#hero { text-align: center; color: #f3f6ff; font-size: 34px; font-weight: 800; margin: 8px 0 2px; }
#subhero { text-align: center; color: #89a2df; font-size: 12px; margin-bottom: 14px; letter-spacing: 0.12em; text-transform: uppercase; }
.panel, .metric-card, .stage-card .wrap, .result-card .wrap, .info-chip, .score-chip {
    background: rgba(11,16,35,0.86) !important;
    border: 1px solid rgba(83,109,205,0.18) !important;
    box-shadow: 0 10px 40px rgba(0,0,0,0.35), inset 0 0 0 1px rgba(255,255,255,0.03);
    border-radius: 16px !important;
}
.info-chip { padding: 10px 12px !important; min-height: 72px; }
.info-chip-label, .score-chip-label { color: #9cb2ef; font-size: 10px; text-transform: uppercase; letter-spacing: 0.1em; }
.info-chip-value { color: #f5f7ff; font-size: 14px; font-weight: 700; margin-top: 8px; line-height: 1.3; }
.metric-card { padding: 12px 14px !important; min-height: 88px; }
.metric-label { font-size: 10px; text-transform: uppercase; letter-spacing: 0.1em; color: #6e7fae; }
.metric-value { font-size: 24px; line-height: 1; font-weight: 800; margin-top: 8px; color: #f3f7ff; }
.metric-sub { margin-top: 8px; color: #7d90c0; font-size: 11px; }
.score-chip { padding: 10px 12px !important; min-height: 84px; border-width: 1px !important; }
.score-chip-value { font-size: 21px; font-weight: 900; margin-top: 8px; }
.score-row-title { color: #d7e3ff; font-size: 13px; font-weight: 700; margin: 6px 0 8px; }
.accent-cyan .metric-value { color: #4de2d6; }
.accent-blue .metric-value { color: #70a6ff; }
.accent-gold .metric-value { color: #ffbe55; }
.accent-magenta .metric-value { color: #de8cff; }
.score-cyan { border-color: rgba(77,226,214,0.35) !important; background: linear-gradient(180deg, rgba(14,41,45,0.95), rgba(10,22,30,0.92)) !important; }
.score-cyan .score-chip-value { color: #4de2d6; }
.score-blue { border-color: rgba(112,166,255,0.35) !important; background: linear-gradient(180deg, rgba(15,28,56,0.95), rgba(10,18,36,0.92)) !important; }
.score-blue .score-chip-value { color: #70a6ff; }
.score-green { border-color: rgba(92,238,141,0.35) !important; background: linear-gradient(180deg, rgba(15,44,29,0.95), rgba(10,24,20,0.92)) !important; }
.score-green .score-chip-value { color: #5cee8d; }
.score-gold { border-color: rgba(255,190,85,0.35) !important; background: linear-gradient(180deg, rgba(58,40,16,0.95), rgba(30,20,10,0.92)) !important; }
.score-gold .score-chip-value { color: #ffbe55; }
.score-pink { border-color: rgba(222,140,255,0.35) !important; background: linear-gradient(180deg, rgba(45,20,52,0.95), rgba(22,12,30,0.92)) !important; }
.score-pink .score-chip-value { color: #de8cff; }
button.primary-btn { background: linear-gradient(180deg, #2ae6cb, #109e92) !important; border: 0 !important; color: #04111a !important; font-weight: 800 !important; }
button.secondary-btn { background: rgba(20,31,66,0.92) !important; border: 1px solid rgba(96,130,240,0.22) !important; color: #d9e4ff !important; }
button.reset-btn { background: linear-gradient(180deg, #ff8a7a, #db5236) !important; border: 0 !important; color: white !important; font-weight: 800 !important; }
.gradio-image, .gr-image { min-height: 220px !important; }
.gr-image .upload-container, .gr-image .empty {
    background: linear-gradient(180deg, rgba(18,24,52,0.96), rgba(8,12,28,0.98)) !important;
    border: 1.5px dashed rgba(92,130,255,0.55) !important;
    border-radius: 16px !important;
}
.gr-image .upload-text, .gr-image .or, .gr-image .upload-text span { color: #dce8ff !important; font-weight: 700 !important; }
video, .source-selection, button[aria-label='Webcam'], button[aria-label='Take Photo'] { display: none !important; }
"""

with gr.Blocks(title="Urban Change Detection Dashboard", css=CSS, theme=gr.themes.Base()) as demo:
    gr.Markdown("<div id='hero'>Urban Change Detection System</div>")
    gr.Markdown("<div id='subhero'>Threshold-aware dashboard</div>")

    with gr.Row():
        model_dropdown = gr.Radio(label="Select Model", choices=list(MODEL_CONFIGS.keys()), value=DEFAULT_MODEL_NAME)

    def _info_card(label, value):
        return f"<div class='info-chip'><div class='info-chip-label'>{label}</div><div class='info-chip-value'>{value}</div></div>"

    with gr.Row():
        model_type_box = gr.HTML(_info_card("Model Type", _model_info(DEFAULT_MODEL_NAME)["Model Type"]))
        backbone_box = gr.HTML(_info_card("Backbone", _model_info(DEFAULT_MODEL_NAME)["Backbone"]))
        task_box = gr.HTML(_info_card("Task", _model_info(DEFAULT_MODEL_NAME)["Task"]))
        dataset_box = gr.HTML(_info_card("Dataset", _model_info(DEFAULT_MODEL_NAME)["Dataset"]))
        best_threshold_box = gr.HTML(_info_card("Best Threshold", "Auto"))

    with gr.Row():
        threshold_slider = gr.Slider(0.1, 0.9, value=0.5, step=0.1, label="Selected Threshold")

    with gr.Row():
        img_t1_in = gr.Image(label="Image T1 · Before", type="pil", image_mode="RGB", height=240, elem_classes=["stage-card"], sources=["upload"])
        img_t2_in = gr.Image(label="Image T2 · After", type="pil", image_mode="RGB", height=240, elem_classes=["stage-card"], sources=["upload"])
        gt_mask_in = gr.Image(label="Ground Truth Mask (optional)", type="pil", image_mode="L", height=240, elem_classes=["stage-card"], sources=["upload"])

    with gr.Row():
        run_btn = gr.Button("Detect Changes", elem_classes=["primary-btn"])
        clear_btn = gr.Button("Load Demo / Clear", elem_classes=["secondary-btn"])

    with gr.Row():
        bbox_out = gr.Image(label="Bounding Boxes", type="numpy", interactive=False, height=235, elem_classes=["result-card"])
        binary_mask_out = gr.Image(label="Binary Mask", type="numpy", interactive=False, height=235, elem_classes=["result-card"])
        heatmap_out = gr.Image(label="Confidence Heatmap", type="numpy", interactive=False, height=235, elem_classes=["result-card"])
        overlay_out = gr.Image(label="Change Overlay", type="numpy", interactive=False, height=235, elem_classes=["result-card"])

    with gr.Row():
        contour_out = gr.Image(label="Contours", type="numpy", interactive=False, height=220, elem_classes=["result-card"])
        before_after_out = gr.Image(label="T1 vs T2", type="numpy", interactive=False, height=220, elem_classes=["result-card"])
        compare_overlay_out = gr.Image(label="T2 vs Overlay", type="numpy", interactive=False, height=220, elem_classes=["result-card"])
        focus_out = gr.Image(label="Threshold Focus Map", type="numpy", interactive=False, height=220, elem_classes=["result-card"])

    def _metric_card(label, value, sub, color_class):
        return f"<div class='metric-card {color_class}'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div><div class='metric-sub'>{sub}</div></div>"

    with gr.Row():
        change_html = gr.HTML(_metric_card("Change Area", "0.00%", "of total image area", "accent-cyan"))
        conf_html = gr.HTML(_metric_card("Confidence", "0%", "mean score", "accent-blue"))
        region_html = gr.HTML(_metric_card("Detected Regions", "0", "bounding box count", "accent-gold"))
        time_html = gr.HTML(_metric_card("Processing Time", "Idle", "total inference time", "accent-magenta"))

    with gr.Row():
        gr.HTML("<div class='score-row-title'>Selected Threshold Metrics</div>")
    with gr.Row():
        sel_acc = gr.HTML()
        sel_prec = gr.HTML()
        sel_rec = gr.HTML()
        sel_iou = gr.HTML()
        sel_f1 = gr.HTML()

    with gr.Row():
        gr.HTML("<div class='score-row-title'>Best Threshold Metrics</div>")
    with gr.Row():
        best_acc = gr.HTML()
        best_prec = gr.HTML()
        best_rec = gr.HTML()
        best_iou = gr.HTML()
        best_f1 = gr.HTML()

    with gr.Row():
        reset_btn = gr.Button("Reset", elem_classes=["reset-btn"])

    def _score_card(label, value, color_class):
        return f"<div class='score-chip {color_class}'><div class='score-chip-label'>{label}</div><div class='score-chip-value'>{value}</div></div>"

    def _metric_value(metrics, key):
        value = metrics.get(key)
        return "N/A" if value is None else f"{value:.4f}"

    def _metrics_row(metrics):
        return (
            _score_card("Accuracy", _metric_value(metrics, "accuracy"), "score-cyan"),
            _score_card("Precision", _metric_value(metrics, "precision"), "score-blue"),
            _score_card("Recall", _metric_value(metrics, "recall"), "score-green"),
            _score_card("IoU", _metric_value(metrics, "iou"), "score-gold"),
            _score_card("F1", _metric_value(metrics, "f1"), "score-pink"),
        )

    def _update_model_boxes(model_name, best_thr_text="Auto"):
        info = _model_info(model_name)
        return (
            _info_card("Model Type", info["Model Type"]),
            _info_card("Backbone", info["Backbone"]),
            _info_card("Task", info["Task"]),
            _info_card("Dataset", info["Dataset"]),
            _info_card("Best Threshold", best_thr_text),
        )

    def _run(img_t1, img_t2, gt_mask, model_name, threshold):
        result = run_dashboard(img_t1, img_t2, gt_mask, model_name, threshold)
        info_cards = _update_model_boxes(model_name, f"{result['best_threshold']:.2f}")
        selected_row = _metrics_row(result["selected_metrics"])
        best_row = _metrics_row(result["best_metrics"])
        return (
            info_cards[0], info_cards[1], info_cards[2], info_cards[3], info_cards[4],
            result["bbox_image"], result["binary_mask"], result["prob_vis"], result["overlay_pred"],
            result["contour_image"], result["before_after_image"], result["compare_overlay_image"], result["confidence_focus_image"],
            _metric_card("Change Area", result["change_pct"], "of total image area", "accent-cyan"),
            _metric_card("Confidence", result["conf_pct"], "mean score", "accent-blue"),
            _metric_card("Detected Regions", result["regions"], "bounding box count", "accent-gold"),
            _metric_card("Processing Time", result["infer_time"], "total inference time", "accent-magenta"),
            selected_row[0], selected_row[1], selected_row[2], selected_row[3], selected_row[4],
            best_row[0], best_row[1], best_row[2], best_row[3], best_row[4],
        )

    def _clear(model_name):
        info_cards = _update_model_boxes(model_name, "Auto")
        empty = {"accuracy": None, "precision": None, "recall": None, "iou": None, "f1": None}
        selected_row = _metrics_row(empty)
        best_row = _metrics_row(empty)
        return (
            info_cards[0], info_cards[1], info_cards[2], info_cards[3], info_cards[4],
            None, None, None, None,
            None, None, None, None,
            _metric_card("Change Area", "0.00%", "of total image area", "accent-cyan"),
            _metric_card("Confidence", "0%", "mean score", "accent-blue"),
            _metric_card("Detected Regions", "0", "bounding box count", "accent-gold"),
            _metric_card("Processing Time", "Idle", "total inference time", "accent-magenta"),
            selected_row[0], selected_row[1], selected_row[2], selected_row[3], selected_row[4],
            best_row[0], best_row[1], best_row[2], best_row[3], best_row[4],
        )

    model_dropdown.change(
        fn=lambda model_name: _update_model_boxes(model_name, "Auto"),
        inputs=model_dropdown,
        outputs=[model_type_box, backbone_box, task_box, dataset_box, best_threshold_box],
    )

    run_btn.click(
        fn=_run,
        inputs=[img_t1_in, img_t2_in, gt_mask_in, model_dropdown, threshold_slider],
        outputs=[
            model_type_box, backbone_box, task_box, dataset_box, best_threshold_box,
            bbox_out, binary_mask_out, heatmap_out, overlay_out,
            contour_out, before_after_out, compare_overlay_out, focus_out,
            change_html, conf_html, region_html, time_html,
            sel_acc, sel_prec, sel_rec, sel_iou, sel_f1,
            best_acc, best_prec, best_rec, best_iou, best_f1,
        ],
    )

    clear_btn.click(
        fn=_clear,
        inputs=[model_dropdown],
        outputs=[
            model_type_box, backbone_box, task_box, dataset_box, best_threshold_box,
            bbox_out, binary_mask_out, heatmap_out, overlay_out,
            contour_out, before_after_out, compare_overlay_out, focus_out,
            change_html, conf_html, region_html, time_html,
            sel_acc, sel_prec, sel_rec, sel_iou, sel_f1,
            best_acc, best_prec, best_rec, best_iou, best_f1,
        ],
    )

    reset_btn.click(
        fn=lambda model_name: (
            model_name,
            _info_card("Model Type", _model_info(model_name)["Model Type"]),
            _info_card("Backbone", _model_info(model_name)["Backbone"]),
            _info_card("Task", _model_info(model_name)["Task"]),
            _info_card("Dataset", _model_info(model_name)["Dataset"]),
            _info_card("Best Threshold", "Auto"),
            0.5,
            None, None, None,
            None, None, None, None,
            None, None, None, None,
            _metric_card("Change Area", "0.00%", "of total image area", "accent-cyan"),
            _metric_card("Confidence", "0%", "mean score", "accent-blue"),
            _metric_card("Detected Regions", "0", "bounding box count", "accent-gold"),
            _metric_card("Processing Time", "Idle", "total inference time", "accent-magenta"),
            *_metrics_row({"accuracy": None, "precision": None, "recall": None, "iou": None, "f1": None}),
            *_metrics_row({"accuracy": None, "precision": None, "recall": None, "iou": None, "f1": None}),
        ),
        inputs=[model_dropdown],
        outputs=[
            model_dropdown,
            model_type_box, backbone_box, task_box, dataset_box, best_threshold_box,
            threshold_slider,
            img_t1_in, img_t2_in, gt_mask_in,
            bbox_out, binary_mask_out, heatmap_out, overlay_out,
            contour_out, before_after_out, compare_overlay_out, focus_out,
            change_html, conf_html, region_html, time_html,
            sel_acc, sel_prec, sel_rec, sel_iou, sel_f1,
            best_acc, best_prec, best_rec, best_iou, best_f1,
        ],
    )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)

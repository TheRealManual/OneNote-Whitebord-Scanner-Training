"""
JSON schema-style validation helpers for project output files.

Lightweight checks — no external schema library required.
"""


TRAINING_HISTORY_REQUIRED_KEYS = {
    'train_loss',
    'val_loss',
    'val_iou',
    'val_f1',
    'config',
    'results',
    'epoch_times',
}

TRAINING_HISTORY_CONFIG_REQUIRED_KEYS = {
    'model',
    'num_classes',
    'epochs',
    'batch_size',
    'learning_rate',
    'optimizer',
    'weight_decay',
    'loss_type',
    'loss_function',
    'scheduler',
    'warmup_epochs',
    'patience',
    'img_height',
    'img_width',
    'img_resolution',
    'num_train_images',
    'num_val_images',
    'use_amp',
    'device',
    'seed',
    'deterministic',
    'training_start_time',
    'data_dir',
}

TRAINING_HISTORY_RESULTS_REQUIRED_KEYS = {
    'best_val_loss',
    'best_val_f1',
    'best_val_iou',
    'final_epoch',
    'total_training_time_seconds',
    'early_stopped',
}

COMPARISON_SUMMARY_REQUIRED_KEYS = {
    'iou',
    'f1',
    'precision',
    'recall',
    'pixel_accuracy',
    'dice',
    'edge_iou',
}

BASELINE_RESULTS_REQUIRED_KEYS = {
    'method',
    'results',
}

BASELINE_METRIC_KEYS = {
    'iou',
    'f1',
    'precision',
    'recall',
    'pixel_accuracy',
    'dice',
    'edge_iou',
    'boundary_f1',
}


def validate_training_history(data: dict) -> list:
    """Validate a training_history.json dict. Returns list of error strings."""
    errors = []
    missing_top = TRAINING_HISTORY_REQUIRED_KEYS - set(data.keys())
    if missing_top:
        errors.append(f"Missing top-level keys: {missing_top}")

    if 'config' in data and isinstance(data['config'], dict):
        missing_cfg = TRAINING_HISTORY_CONFIG_REQUIRED_KEYS - set(data['config'].keys())
        if missing_cfg:
            errors.append(f"Missing config keys: {missing_cfg}")
    elif 'config' in TRAINING_HISTORY_REQUIRED_KEYS:
        errors.append("'config' is not a dict")

    if 'results' in data and isinstance(data['results'], dict):
        missing_res = TRAINING_HISTORY_RESULTS_REQUIRED_KEYS - set(data['results'].keys())
        if missing_res:
            errors.append(f"Missing results keys: {missing_res}")
    elif 'results' in TRAINING_HISTORY_REQUIRED_KEYS:
        errors.append("'results' is not a dict")

    return errors


def validate_comparison_summary(data: dict) -> list:
    """Validate a comparison_summary.json dict. Returns list of error strings."""
    errors = []
    missing = COMPARISON_SUMMARY_REQUIRED_KEYS - set(data.keys())
    if missing:
        errors.append(f"Missing keys: {missing}")
    return errors


def validate_baseline_results(data: dict) -> list:
    """Validate classical baseline results JSON. Returns list of error strings."""
    errors = []
    missing = BASELINE_RESULTS_REQUIRED_KEYS - set(data.keys())
    if missing:
        errors.append(f"Missing top-level keys: {missing}")
    return errors

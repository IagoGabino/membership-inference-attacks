"""
Central experiment configuration.

Final runners use the named dictionaries near the bottom of this file. Legacy
aliases are kept so older scripts still import successfully, but shadow/HAR/
Sensorless/Canary are intentionally absent from the final experiment maps.
"""

from __future__ import annotations

from copy import deepcopy


# ============================================================
# 1. Global settings
# ============================================================

DEFAULT_SEEDS = [42, 123, 7]

SMALL = 10_000
MEDIUM = 40_000
LARGE = 100_000

SMALL_20NG = 5_000
MEDIUM_20NG = 15_000
FULL_20NG = None
MAX_TFIDF_FEATURES = 5000

DEFAULT_BATCH_SIZE = 256
DEFAULT_ATTACK_TEST_RATIO = 0.30

FIT_RATIO = 0.50
VAL_RATIO = 0.10
NONMEMBER_RATIO = 0.30
HOLDOUT_RATIO = 0.10

RESULTS_FINAL_DIR = "results_final"

DEBUG_CONFIG = {
    "experiment_sample_size": 800,
    "epochs": 1,
    "batch_size": 128,
    "max_tfidf_features": 1000,
}


def _named(config: dict, name: str) -> dict:
    cfg = deepcopy(config)
    cfg["name"] = name
    return cfg


# ============================================================
# 2. CDC configs
# ============================================================

CDC_BASELINE_CONFIG = {
    "name": "cdc_baseline",
    "hidden_layers": [64, 32],
    "dropout": 0.1,
    "output_dim": 2,
    "epochs": 30,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "early_stopping": True,
    "patience": 7,
    "batch_size": DEFAULT_BATCH_SIZE,
    "regularization_type": "dropout+weight_decay+early_stopping",
    "experiment_sample_size": MEDIUM,
}

CDC_REGULARIZED_CONFIG = {
    "name": "cdc_regularized",
    "hidden_layers": [64, 32],
    "dropout": 0.3,
    "output_dim": 2,
    "epochs": 100,
    "learning_rate": 1e-3,
    "weight_decay": 1e-3,
    "early_stopping": True,
    "patience": 10,
    "batch_size": DEFAULT_BATCH_SIZE,
    "regularization_type": "dropout+weight_decay+early_stopping",
    "experiment_sample_size": MEDIUM,
}

CDC_OVERFITTED_CONFIG = {
    "name": "cdc_overfitted",
    "hidden_layers": [256, 128, 64],
    "dropout": 0.0,
    "output_dim": 2,
    "epochs": 80,
    "learning_rate": 1e-3,
    "weight_decay": 0.0,
    "early_stopping": False,
    "patience": None,
    "batch_size": DEFAULT_BATCH_SIZE,
    "regularization_type": "none",
    "experiment_sample_size": MEDIUM,
    "restore_best_model": False,
}

CDC_VULNERABLE_2K_CONFIG = {
    "name": "cdc_vulnerable_2k",
    "hidden_layers": [512, 256, 128],
    "dropout": 0.0,
    "output_dim": 2,
    "epochs": 201,
    "learning_rate": 1e-3,
    "weight_decay": 0.0,
    "early_stopping": False,
    "patience": None,
    "batch_size": 128,
    "regularization_type": "none_extreme",
    "experiment_sample_size": 2_000,
    "restore_best_model": False,
}

CDC_VULNERABLE_5K_CONFIG = {
    **CDC_VULNERABLE_2K_CONFIG,
    "name": "cdc_vulnerable_5k",
    "experiment_sample_size": 5_000,
}


# ============================================================
# 3. CDC DP noise configs
# ============================================================

CDC_DP_NOISE_BASE_CONFIG = {
    **CDC_VULNERABLE_2K_CONFIG,
    "name": "cdc_dp_noise_base",
    "epochs": 80,
    "learning_rate": 3e-3,
    "optimizer": "adam",
    "use_class_weights": True,
    "regularization_type": "dp_sgd",
    "batch_size": DEFAULT_BATCH_SIZE,
    "max_physical_batch_size": 64,
    "max_grad_norm": 1.0,
    "delta": 1e-5,
    "restore_best_model": False,
}

CDC_DP_NOISE_MULTIPLIERS = [0.1, 0.3, 0.6, 1.0]

CDC_DP_EXPERIMENTS = {
    f"cdc_dp_noise_{str(noise).replace('.', '_')}": {
        **CDC_DP_NOISE_BASE_CONFIG,
        "name": f"cdc_dp_noise_{str(noise).replace('.', '_')}",
        "noise_multiplier": noise,
    }
    for noise in CDC_DP_NOISE_MULTIPLIERS
}


# ============================================================
# 4. 20 Newsgroups configs
# ============================================================

NEWS20_BASELINE_CONFIG = {
    "name": "news20_baseline",
    "hidden_layers": [256, 128],
    "dropout": 0.1,
    "output_dim": 20,
    "epochs": 30,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "early_stopping": True,
    "patience": 7,
    "batch_size": DEFAULT_BATCH_SIZE,
    "regularization_type": "dropout+weight_decay+early_stopping",
    "experiment_sample_size": MEDIUM_20NG,
    "max_tfidf_features": MAX_TFIDF_FEATURES,
}

NEWS20_BASELINE_SMALL_CONFIG = {
    **NEWS20_BASELINE_CONFIG,
    "name": "news20_baseline_small",
    "experiment_sample_size": SMALL_20NG,
}

NEWS20_REGULARIZED_CONFIG = {
    "name": "news20_regularized",
    "hidden_layers": [256, 128],
    "dropout": 0.3,
    "output_dim": 20,
    "epochs": 100,
    "learning_rate": 1e-3,
    "weight_decay": 1e-3,
    "early_stopping": True,
    "patience": 10,
    "batch_size": DEFAULT_BATCH_SIZE,
    "regularization_type": "dropout+weight_decay+early_stopping",
    "experiment_sample_size": MEDIUM_20NG,
    "max_tfidf_features": MAX_TFIDF_FEATURES,
}

NEWS20_OVERFITTED_CONFIG = {
    "name": "news20_overfitted",
    "hidden_layers": [512, 256, 128],
    "dropout": 0.0,
    "output_dim": 20,
    "epochs": 80,
    "learning_rate": 1e-3,
    "weight_decay": 0.0,
    "early_stopping": False,
    "patience": None,
    "batch_size": DEFAULT_BATCH_SIZE,
    "regularization_type": "none",
    "experiment_sample_size": MEDIUM_20NG,
    "restore_best_model": False,
    "max_tfidf_features": MAX_TFIDF_FEATURES,
}

NEWS20_OVERFITTED_SMALL_CONFIG = {
    **NEWS20_OVERFITTED_CONFIG,
    "name": "news20_overfitted_small",
    "experiment_sample_size": SMALL_20NG,
}

NEWS20_VULNERABLE_2K_CONFIG = {
    "name": "news20_vulnerable_2k",
    "hidden_layers": [1024, 512, 256],
    "dropout": 0.0,
    "output_dim": 20,
    "epochs": 200,
    "learning_rate": 1e-3,
    "weight_decay": 0.0,
    "early_stopping": False,
    "patience": None,
    "batch_size": 128,
    "regularization_type": "none_extreme",
    "experiment_sample_size": 2_000,
    "restore_best_model": False,
    "max_tfidf_features": MAX_TFIDF_FEATURES,
}

NEWS20_VULNERABLE_5K_CONFIG = {
    **NEWS20_VULNERABLE_2K_CONFIG,
    "name": "news20_vulnerable_5k",
    "experiment_sample_size": 5_000,
}


# ============================================================
# 5. 20 Newsgroups DP noise configs
# ============================================================

NEWS20_DP_NOISE_BASE_CONFIG = {
    **NEWS20_OVERFITTED_SMALL_CONFIG,
    "name": "news20_dp_noise_base",
    "epochs": 50,
    "max_tfidf_features": 2000,
    "hidden_layers": [512, 256, 128],
    "learning_rate": 1e-3,
    "optimizer": "adam",
    "regularization_type": "dp_sgd",
    "batch_size": 64,
    "max_physical_batch_size": 16,
    "max_grad_norm": 1.0,
    "delta": 1e-5,
    "restore_best_model": False,
}

NEWS20_DP_NOISE_MULTIPLIERS = [0.1, 0.3, 0.6, 1.0]

NEWS20_DP_EXPERIMENTS = {
    f"news20_dp_noise_{str(noise).replace('.', '_')}": {
        **NEWS20_DP_NOISE_BASE_CONFIG,
        "name": f"news20_dp_noise_{str(noise).replace('.', '_')}",
        "noise_multiplier": noise,
    }
    for noise in NEWS20_DP_NOISE_MULTIPLIERS
}


# ============================================================
# 6. API exposure configs
# ============================================================

API_EXPOSURE_MODES = [
    {"name": "oracle_full", "description": "Loss, confidence, true-label features.", "round_decimals": None},
    {"name": "probs_full", "description": "Full probability vector, no true-label loss.", "round_decimals": None},
    {"name": "probs_rounded_2", "description": "Full probability vector rounded to 2 decimals.", "round_decimals": 2},
    {"name": "probs_rounded_1", "description": "Full probability vector rounded to 1 decimal.", "round_decimals": 1},
    {"name": "top1_confidence", "description": "Only maximum confidence exposed.", "round_decimals": None},
    {"name": "label_only", "description": "Predicted label plus true-label correctness.", "round_decimals": None},
]

NEWS20_API_EXPOSURE_EXPERIMENTS = [
    "news20_baseline",
    "news20_overfitted",
    "news20_vulnerable_2k",
    "news20_regularized",
]

CDC_API_EXPOSURE_EXPERIMENTS = [
    "cdc_baseline",
    "cdc_overfitted",
    "cdc_regularized",
]


# ============================================================
# 7. Top-k configs
# ============================================================

TOPK_PERCENTAGES = [0.01, 0.05, 0.10]

NEWS20_TOPK_RISK_EXPERIMENTS = [
    "news20_baseline",
    "news20_baseline_small",
    "news20_regularized",
    "news20_overfitted",
    "news20_overfitted_small",
    "news20_vulnerable_2k",
    "news20_vulnerable_5k",
]

CDC_TOPK_RISK_EXPERIMENTS = [
    "cdc_baseline",
    "cdc_regularized",
    "cdc_overfitted",
    "cdc_vulnerable_2k",
    "cdc_vulnerable_5k",
]


# ============================================================
# 8. Final experiment dictionaries
# ============================================================

CDC_EXPERIMENTS = {
    "cdc_baseline": CDC_BASELINE_CONFIG,
    "cdc_regularized": CDC_REGULARIZED_CONFIG,
    "cdc_overfitted": CDC_OVERFITTED_CONFIG,
    "cdc_vulnerable_2k": CDC_VULNERABLE_2K_CONFIG,
    "cdc_vulnerable_5k": CDC_VULNERABLE_5K_CONFIG,
}

NEWS20_EXPERIMENTS = {
    "news20_baseline": NEWS20_BASELINE_CONFIG,
    "news20_baseline_small": NEWS20_BASELINE_SMALL_CONFIG,
    "news20_regularized": NEWS20_REGULARIZED_CONFIG,
    "news20_overfitted": NEWS20_OVERFITTED_CONFIG,
    "news20_overfitted_small": NEWS20_OVERFITTED_SMALL_CONFIG,
    "news20_vulnerable_2k": NEWS20_VULNERABLE_2K_CONFIG,
    "news20_vulnerable_5k": NEWS20_VULNERABLE_5K_CONFIG,
}

NEWS20_TOPK_EXPERIMENTS = {
    name: NEWS20_EXPERIMENTS[name] for name in NEWS20_TOPK_RISK_EXPERIMENTS
}

ALL_FINAL_EXPERIMENTS = {
    **CDC_EXPERIMENTS,
    **CDC_DP_EXPERIMENTS,
    **NEWS20_EXPERIMENTS,
    **NEWS20_DP_EXPERIMENTS,
}


# ============================================================
# 9. Legacy compatibility aliases
# ============================================================

# Older scripts import these short names. They are aliases to the final CDC
# configs where possible, so old code still works without joining the final
# pipeline to shadow/HAR/Sensorless/Canary.
BASELINE_CONFIG = _named(CDC_BASELINE_CONFIG, "baseline")
REGULARIZED_CONFIG = _named(CDC_REGULARIZED_CONFIG, "regularized")
OVERFITTED_CONFIG = _named(CDC_OVERFITTED_CONFIG, "overfitted")
BASELINE_SMALL_CONFIG = {**BASELINE_CONFIG, "name": "baseline_small", "experiment_sample_size": SMALL}
OVERFITTED_SMALL_CONFIG = {**OVERFITTED_CONFIG, "name": "overfitted_small", "experiment_sample_size": SMALL}
BASELINE_LARGE_CONFIG = {**BASELINE_CONFIG, "name": "baseline_large", "experiment_sample_size": LARGE}
VULNERABLE_2K_CONFIG = _named(CDC_VULNERABLE_2K_CONFIG, "vulnerable_2k")
VULNERABLE_5K_CONFIG = _named(CDC_VULNERABLE_5K_CONFIG, "vulnerable_5k")

DP_CONFIG = {
    **CDC_DP_NOISE_BASE_CONFIG,
    "name": "dp_sgd",
    "hidden_layers": [64, 32],
    "epochs": 20,
    "learning_rate": 1e-3,
    "experiment_sample_size": MEDIUM,
    "noise_multiplier": 1.0,
}

NOISE_MULTIPLIERS = CDC_DP_NOISE_MULTIPLIERS
DP_NOISE_CONFIGS = {
    key.replace("cdc_", ""): _named(value, key.replace("cdc_", ""))
    for key, value in CDC_DP_EXPERIMENTS.items()
}

ALL_CONFIGS = {
    "baseline": BASELINE_CONFIG,
    "overfitted": OVERFITTED_CONFIG,
    "regularized": REGULARIZED_CONFIG,
    "baseline_small": BASELINE_SMALL_CONFIG,
    "overfitted_small": OVERFITTED_SMALL_CONFIG,
    "baseline_large": BASELINE_LARGE_CONFIG,
    "dp_sgd": DP_CONFIG,
    "vulnerable_2k": VULNERABLE_2K_CONFIG,
    "vulnerable_5k": VULNERABLE_5K_CONFIG,
    **DP_NOISE_CONFIGS,
    **CDC_EXPERIMENTS,
    **CDC_DP_EXPERIMENTS,
    **NEWS20_EXPERIMENTS,
    **NEWS20_DP_EXPERIMENTS,
}

MVP_EXPERIMENTS = ["baseline", "overfitted", "regularized"]
MAIN_EXPERIMENTS = ["baseline", "overfitted", "regularized"]
SIZE_EXPERIMENTS = ["baseline_small", "overfitted_small", "baseline_large"]
VULNERABLE_EXPERIMENTS = ["vulnerable_2k", "vulnerable_5k"]
DP_NOISE_EXPERIMENTS = list(DP_NOISE_CONFIGS.keys())

API_EXPOSURE_EXPERIMENTS = CDC_API_EXPOSURE_EXPERIMENTS
API_EXPOSURE_STRESS_EXPERIMENTS = [
    "cdc_baseline",
    "cdc_overfitted",
    "cdc_regularized",
    "cdc_vulnerable_2k",
    "cdc_vulnerable_5k",
]

LEGACY_EXPERIMENTS_DO_NOT_RUN = {
    "shadow_models": "Kept in src/shadow_attack.py and main_experiment.py for report-only negative results.",
    "har": "No final runner imports HAR configs.",
    "sensorless": "No final runner imports Sensorless configs.",
    "canary": "No final runner imports Canary configs.",
}

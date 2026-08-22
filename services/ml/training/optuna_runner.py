"""Optuna hyperparameter search for Pix2Pix GAN training.

Runs N trials of ``train_gan`` with different hyperparameter combinations
and logs the best configuration to MLflow. Uses TPE (Tree-structured
Parzen Estimator) sampler for efficient search over:

- Generator learning rate (log scale: 1e-5 to 1e-3)
- Discriminator LR factor (0.25 to 1.0)
- Batch size (8, 16, 32)
- Loss weights: pixel (50-200), perceptual (0-10), adversarial (0.5-5),
  physics (0-10)

Each trial runs a short training (default 5 epochs) and uses validation
SSIM as the optimization objective (higher = better).

Usage:
    python -m services.ml.training.optuna_runner \\
        --n-trials 20 --epochs-per-trial 5

Reference:
    Akiba, T. et al. (2019). "Optuna: A Next-generation Hyperparameter
    Optimization Framework." KDD.
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import mlflow
import optuna
from omegaconf import DictConfig, OmegaConf

# Proje kökünü path'e ekle (script doğrudan çalıştırılabilsin)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.ml.training.gan_train import train_gan

# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------
SEARCH_SPACE = {
    "learning_rate": ("log_uniform", 1e-5, 1e-3),
    "discriminator_lr_factor": ("uniform", 0.25, 1.0),
    "batch_size": ("categorical", [8, 16, 32]),
    "pixel_weight": ("uniform", 50.0, 200.0),
    "perceptual_weight": ("uniform", 0.0, 10.0),
    "adversarial_weight": ("uniform", 0.5, 5.0),
    "physics_weight": ("uniform", 0.0, 10.0),
}


def _suggest(trial: optuna.Trial, name: str, spec: tuple) -> float | int:
    """Suggest a hyperparameter value according to its spec."""
    kind, *args = spec
    if kind == "log_uniform":
        return trial.suggest_float(name, *args, log=True)
    if kind == "uniform":
        return trial.suggest_float(name, *args)
    if kind == "categorical":
        return trial.suggest_categorical(name, args[0])
    raise ValueError(f"Unknown search space kind: {kind}")


def _apply_overrides(cfg: DictConfig, params: dict) -> DictConfig:
    """Apply trial hyperparameters to a copy of the base config."""
    cfg = deepcopy(cfg)
    # Defaults merge edilmiş olmalı (compose kullanıldıysa)
    if "training" not in cfg:
        raise ValueError(
            "Config 'training' key missing — defaults not merged. "
            "Use hydra.compose() instead of OmegaConf.load()."
        )
    cfg.training.learning_rate = params["learning_rate"]
    cfg.training.discriminator_lr_factor = params["discriminator_lr_factor"]
    cfg.training.batch_size = params["batch_size"]
    cfg.loss.weights.pixel = params["pixel_weight"]
    cfg.loss.weights.perceptual = params["perceptual_weight"]
    cfg.loss.weights.adversarial = params["adversarial_weight"]
    cfg.loss.weights.physics = params["physics_weight"]
    # Her trial kendi MLflow run'ını açar
    cfg.mlflow.run_name = f"optuna-trial-{params.get('_trial_number', 0)}"
    return cfg


def _extract_val_ssim(checkpoint_dir: Path) -> float:
    """Read the last validation SSIM from the JSONL log."""
    jsonl_path = checkpoint_dir / "validation_results.jsonl"
    if not jsonl_path.exists():
        return 0.0
    last_ssim = 0.0
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                last_ssim = float(record.get("ssim", 0.0))
            except json.JSONDecodeError:
                continue
    return last_ssim


def objective(
    trial: optuna.Trial,
    base_cfg: DictConfig,
    epochs_per_trial: int,
    output_root: Path,
) -> float:
    """Single Optuna trial: train GAN with suggested params, return val SSIM."""
    params = {name: _suggest(trial, name, spec) for name, spec in SEARCH_SPACE.items()}
    params["_trial_number"] = trial.number

    trial_cfg = _apply_overrides(base_cfg, params)
    trial_cfg.training.epochs = epochs_per_trial
    trial_cfg.paths.output_dir = str(output_root / f"trial_{trial.number:03d}")

    print(f"\n{'='*60}")
    print(f"Trial {trial.number}: {params}")
    print(f"{'='*60}")

    try:
        train_gan(trial_cfg)
        val_ssim = _extract_val_ssim(Path(trial_cfg.paths.output_dir))
    except Exception as exc:  # pragma: no cover - eğitim hataları
        print(f"Trial {trial.number} failed: {exc}")
        # Optuna'a "kötü" değer döndür (prune et)
        raise optuna.TrialPruned(f"Trial {trial.number} crashed: {exc}") from exc

    # MLflow'a trial parametrelerini log'la
    with mlflow.start_run(nested=True, run_name=f"optuna-trial-{trial.number}"):
        mlflow.log_params({k: v for k, v in params.items() if not k.startswith("_")})
        mlflow.log_metric("val_ssim", val_ssim)
        mlflow.log_metric("trial_number", trial.number)

    return val_ssim


def run_optuna_search(
    base_cfg: DictConfig,
    n_trials: int = 20,
    epochs_per_trial: int = 5,
    output_root: Path | str = "outputs/optuna",
    study_name: str = "pix2pix-hparam-search",
) -> optuna.Study:
    """Run Optuna hyperparameter search.

    Args:
        base_cfg: Base Hydra config (will be copied per trial).
        n_trials: Number of trials to run.
        epochs_per_trial: Epochs per trial (keep small for speed).
        output_root: Where to save per-trial checkpoints.
        study_name: Optuna study name.

    Returns:
        Completed Optuna Study object.
    """
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    sampler = optuna.samplers.TPESampler(seed=int(base_cfg.seed))
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",  # SSIM'i maksimize et
        sampler=sampler,
    )

    # Parent MLflow run
    mlflow.set_tracking_uri(base_cfg.mlflow.tracking_uri)
    mlflow.set_experiment(f"{base_cfg.mlflow.experiment_name}-optuna")

    with mlflow.start_run(run_name=f"optuna-parent-{study_name}"):
        mlflow.log_params(
            {
                "n_trials": n_trials,
                "epochs_per_trial": epochs_per_trial,
                "study_name": study_name,
            }
        )

        study.optimize(
            lambda trial: objective(trial, base_cfg, epochs_per_trial, output_root),
            n_trials=n_trials,
            show_progress_bar=False,
        )

        # Best params → MLflow
        best_params = {k: v for k, v in study.best_params.items()}
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("best_val_ssim", study.best_value)

        # Best params → JSON dosyası
        best_path = output_root / "best_params.json"
        with best_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "best_value": study.best_value,
                    "best_params": best_params,
                    "n_trials": n_trials,
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )
        mlflow.log_artifact(str(best_path))

        print(f"\n{'='*60}")
        print(f"Best trial: #{study.best_trial.number}")
        print(f"Best val SSIM: {study.best_value:.4f}")
        print(f"Best params: {best_params}")
        print(f"Saved to: {best_path}")
        print(f"{'='*60}")

    return study


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter search for Pix2Pix GAN",
    )
    parser.add_argument("--n-trials", type=int, default=20, help="Number of trials")
    parser.add_argument(
        "--epochs-per-trial",
        type=int,
        default=5,
        help="Epochs per trial (keep small for speed)",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs/optuna",
        help="Where to save per-trial checkpoints",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="pix2pix-hparam-search",
        help="Optuna study name",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point: load base config and run Optuna search."""
    args = parse_args()

    # Base config'i yükle (Hydra olmadan, doğrudan YAML)
    from hydra import compose, initialize_config_dir

    config_dir = PROJECT_ROOT / "services" / "ml" / "conf"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        base_cfg = compose(config_name="config")

    run_optuna_search(
        base_cfg=base_cfg,
        n_trials=args.n_trials,
        epochs_per_trial=args.epochs_per_trial,
        output_root=args.output_root,
        study_name=args.study_name,
    )


if __name__ == "__main__":
    main()

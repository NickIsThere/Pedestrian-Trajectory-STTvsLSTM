from __future__ import annotations

import sys
from pathlib import Path

from data.reader import get_dataset

from utils.train.config import TRAINING_DEFAULTS, _configure_social_lstm_environment, model_config
from utils.train.constants import DOMAIN_VALUES, SAMPLE_MODES, SampleMode
from utils.train.evaluation import evaluate_split_for_model
from utils.train.models.registry import get_model_spec
from utils.train.orchestration import run_training_folds
from utils.train.splits import (
    all_dataset_sequences,
    build_supervised_splits,
    sequence_has_gt,
    split_from_sequences,
)

__all__ = [
    "DOMAIN_VALUES",
    "SAMPLE_MODES",
    "SampleMode",
    "all_dataset_sequences",
    "build_supervised_splits",
    "evaluate_split_for_model",
    "get_model_spec",
    "model_config",
    "run_training_folds",
    "sequence_has_gt",
    "split_from_sequences",
]


def main() -> None:
    '''
    Nick Grebe i6377605
    '''
    train_config = TRAINING_DEFAULTS

    if train_config.model == "lstm":
        _configure_social_lstm_environment(train_config)
        try:
            from App.benchmarks.social_lstm import train as sl_train
        except Exception:
            repo_root = Path(__file__).resolve().parents[1]
            repo_root_str = str(repo_root)
            if repo_root_str not in sys.path:
                sys.path.insert(0, repo_root_str)
            from App.benchmarks.social_lstm import train as sl_train

        sl_train.train()
        return

    dataset = get_dataset("MOT20")
    run_dir = run_training_folds(
        dataset=dataset,
        model_spec=get_model_spec(train_config.model),
        config=model_config,
        output_dir=train_config.output_dir,
        run_id=train_config.run_id,
        leave_one_out=train_config.leave_one_out,
        fold_sequence=train_config.fold_sequence,
        train_steps=train_config.train_steps,
        val_interval=train_config.val_interval,
        batch_size=train_config.batch_size,
        max_val_samples=train_config.max_val_samples,
        seed=train_config.seed,
        training_input_mode=train_config.train_input_mode,
        debug_smoke=train_config.debug_smoke,
        export_evaluation_units=train_config.export_evaluation_units,
        model_family=train_config.model_family,
        model_variant=train_config.model_variant,
        train_domain=train_config.train_domain,
        test_domain=train_config.test_domain,
        synthetic_size_fraction=train_config.synthetic_size_fraction,
    )
    print(f"Training finished. Outputs saved to {run_dir}")


if __name__ == "__main__":
    main()

import os
import time
import csv
import math
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from class_based_fixed_speech_brain import EmotionRecognitionTrainer
from numpy.random import default_rng
from statics import SEED

import random
import numpy as np

# ---------------------
# Repro
# ---------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ---------------------
# Trade-off knobs (adjust freely)
# ---------------------
# TIME_TARGET: seconds you consider "good enough" per trial.
# BETA: how much to penalize time vs. (1 - BCA). Try 0.1–0.4. Higher = more aggressive on time.
BETA = float(os.getenv("HPO_BETA", "0.25"))
TIME_TARGET = float(os.getenv("HPO_TIME_TARGET", "60"))

# ---------------------
# Data
# ---------------------
df = pd.read_parquet("hf://datasets/renumics/emodb/data/train-00000-of-00001-cf0d4b1ae18136ff.parquet")
label_encoder_obj = LabelEncoder()
df["emotion"] = label_encoder_obj.fit_transform(df["emotion"])
mapping = dict(zip(label_encoder_obj.classes_, label_encoder_obj.transform(label_encoder_obj.classes_)))
print("Label mapping:", mapping)

train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)

batch_size = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------
# Search space
# ---------------------
space = {
    'lr': hp.loguniform('lr', math.log(1e-6), math.log(1e-3)),
    'num_epochs': hp.quniform('num_epochs', 1, 10, 1),
    'unfreeze_epoch': hp.quniform('unfreeze_epoch', 0, 5, 1),
    'max_length': hp.choice('max_length', [2 * 16000, 3 * 16000, 4 * 16000, 5 * 16000, 7 * 16000, 10 * 16000])
}

# ---------------------
# Objective
# ---------------------
def objective(params):
    params['num_epochs'] = int(params['num_epochs'])
    params['unfreeze_epoch'] = int(params['unfreeze_epoch'])

    config = {
        "batch_size": batch_size,
        "lr": params['lr'],
        "num_epochs": params['num_epochs'],
        "unfreeze_epoch": params['unfreeze_epoch'],
        "max_length": params['max_length'],
        "device": device
    }
    print(f"\nRunning trial with config: {config}")

    trainer = EmotionRecognitionTrainer(config, train_df, valid_df, mapping)

    start_time = time.time()
    results = trainer.train()
    total_time = time.time() - start_time
    results["total_time"] = float(total_time)

    # Treat validation_accuracy as BCA \in [0,1]
    bca = float(results.get("validation_accuracy", 0.0))

    # Scalarization: minimize loss = (1 - bca) + BETA * log1p(time / TIME_TARGET)
    # log1p keeps penalty tame & smooth; if you prefer linear, use total_time / TIME_TARGET
    time_penalty = math.log1p(total_time / TIME_TARGET)
    loss = (1.0 - bca) + BETA * time_penalty

    # Robust CSV path (works in scripts & notebooks)
    script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    csv_file = os.path.join(script_dir, "hpo_results_hyperopt_bca_1_test.csv")

    fieldnames = [
        "lr", "num_epochs", "unfreeze_epoch", "max_length",
        "validation_accuracy", "total_time", "loss", "time_penalty", "BETA", "TIME_TARGET"
    ]
    file_exists = os.path.exists(csv_file)
    with open(csv_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "lr": config.get("lr"),
            "num_epochs": config.get("num_epochs"),
            "unfreeze_epoch": config.get("unfreeze_epoch"),
            "max_length": config.get("max_length"),
            "validation_accuracy": bca,
            "total_time": total_time,
            "loss": loss,
            "time_penalty": time_penalty,
            "BETA": BETA,
            "TIME_TARGET": TIME_TARGET,
        })

    return {
        'loss': loss,
        'status': STATUS_OK,
        'config': config,
        'results': results
    }

# ---------------------
# Pareto utilities (post-hoc inspection)
# ---------------------
def _pareto_indices(items):
    """
    items: list of dicts with keys 'bca' (higher better), 'time' (lower better)
    returns indices of non-dominated (Pareto-optimal) points.
    """
    idxs = []
    for i, a in enumerate(items):
        dominated = False
        for j, b in enumerate(items):
            if i == j:
                continue
            if (b['bca'] >= a['bca'] and b['time'] <= a['time']) and (b['bca'] > a['bca'] or b['time'] < a['time']):
                dominated = True
                break
        if not dominated:
            idxs.append(i)
    return idxs

def _print_pareto(trials):
    items = []
    for t in trials.results:
        bca = float(t['results'].get('validation_accuracy', 0.0))
        tt  = float(t['results'].get('total_time', 0.0))
        items.append({'bca': bca, 'time': tt, 'config': t['config']})

    if not items:
        print("No trials collected for Pareto analysis.")
        return

    idxs = _pareto_indices(items)
    pareto = [items[i] for i in idxs]
    # sort by time asc, then bca desc
    pareto.sort(key=lambda x: (x['time'], -x['bca']))

    print("\nPareto-optimal configs (maximize BCA, minimize time):")
    for k, p in enumerate(pareto, 1):
        print(f"[{k}] BCA={p['bca']:.4f} | time={p['time']:.2f}s | config={p['config']}")

# ---------------------
# Run Hyperopt
# ---------------------
if __name__ == "__main__":
    trials = Trials()

    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=50,
        trials=trials,
        rstate=default_rng(SEED)
    )

    # Decode max_length choice index back to value
    max_length_options = [2 * 16000, 3 * 16000, 4 * 16000, 5 * 16000, 7 * 16000, 10 * 16000]
    best['max_length'] = max_length_options[best['max_length']]

    print("\nBest hyperparameters by scalarized loss:")
    print(best)

    # Also show Pareto front across all tried configs
    _print_pareto(trials)

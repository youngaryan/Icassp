# SpeechBrain Emotion HPO
https://arxiv.org/abs/2510.07052

> **Hyperparameter optimization (HPO) toolbox for Speech Emotion Recognition (SER) built on [SpeechBrain].**
This repository contains a lightweight pipeline to **prepare emotion datasets**, **train SER models with SpeechBrain**, and **run automated hyperparameter search** ([Ax], [hyperopt], and [Optuna]). It’s intended for reproducible research and quick experimentation.

---

## Highlights

* End‑to‑end workflow: data prep → training → HPO → export best config.
* Reproducible runs with fixed seeds, structured logging, and checkpointing.
* Clear separation between **data preprocessing** and **model/training code**.

---

## Repository structure

```
.
├── data_preprocessing/          # Scripts & utilities
├── speecbrain/                  # (Project package) models, training loops, HPO utilities
├── requirements.txt             # Python dependencies
├── manuscript-ICASSP-2026.pdf   # ICASSP submited manuscript
├── .gitignore
└── README.md
```

---

Install the repo and dependencies:

```bash
# clone
git clone https://github.com/youngaryan/speechbrain-emotion-hpo
cd speechbrain-emotion-hpo

# create venv (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

---

## Example usages


```bash
python ./speechbrain-emotion-hpo/hpo_optimiser_ax.py
```

The pipline will save the best model and generate CSV log file including its expermental result(BCA, Total time...)

[SpeechBrain]: https://github.com/speechbrain/speechbrain
[Optuna]: https://optuna.org/
[Ax]: https://ax.dev/
[hyperopt]: https://github.com/hyperopt/hyperopt

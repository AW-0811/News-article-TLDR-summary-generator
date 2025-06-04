# BART-CNN Summarization with Hyperband Tuning

This project fine-tunes a BART model (`facebook/bart-base`) on the CNN/DailyMail dataset using Optuna's Hyperband pruner for efficient hyperparameter optimization. It also includes an inference script to generate news summaries using the best performing model.

---

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## Training

Run the training with Hyperband optimization:

```bash
python hyperband_train.py
```

- This script loads a subset of the CNN/DailyMail dataset.
- Applies preprocessing and tokenization.
- Uses Optuna with a `HyperbandPruner` to search hyperparameters (batch size, learning rate, etc.).
- Saves the best model in `./bart-cnn-best-so-far/`.

---

## Inference

Use the saved model to summarize validation articles:

```bash
python hyperbandinfer.py
```

This script:
- Loads the best checkpoint from `./bart-cnn-best-so-far/`
- Runs inference on 10 validation samples
- Prints the original article, generated summary, and reference summary

---

## Directory Structure

```
.
├── hyperband_train.py          # Training script with Optuna + Hyperband
├── hyperbandinfer.py           # Inference script
├── requirements.txt            # Dependencies
├── README.md                   # You're reading it
└── bart-cnn-best-so-far/       # Saved best model (created after training)
```

---

## Notes

- Trains on a subset (50K training / 5K validation) for faster hyperparameter tuning.
- To use full dataset, modify the `select(range(...))` lines for both train and validation sets separately.

---

## 📃 License

MIT License

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

Before running inference, **download the best model checkpoint** from this link and place it in your project root as `bart-cnn-best-so-far/`:

[Download Best Model from Google Drive](https://drive.google.com/drive/folders/1A2B3C4D5E6F7G8H9)

Then run:

```bash
python hyperbandinfer.py
```

This script:
- Loads the best checkpoint from `./bart-cnn-best-so-far/`
- Runs inference on 10 validation samples
- Prints the original article, generated summary, and reference summary

---

## Sample Output

See [sample_outputs.log](./sample_outputs.log) for examples of generated summaries.

---

## Directory Structure

```
.
├── hyperband_train.py          # Training script with Optuna + Hyperband
├── hyperbandinfer.py           # Inference script
├── requirements.txt            # Dependencies
├── README.md                   # README file
├── sample_outputs.log          # Logged summaries from inference
└── bart-cnn-best-so-far/       # (Not included; download from Google Drive)
```

---

## Notes

- Trains on a subset (50K training / 5K validation) for faster hyperparameter tuning.
- To use full dataset, modify the `select(range(...))` lines.
- Best model is not included due to file size, download from the link above.

---

## 📃 License

MIT License

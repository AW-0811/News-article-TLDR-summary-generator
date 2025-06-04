# Import necessary libraries
import os
import torch
import optuna
import shutil
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq, EarlyStoppingCallback
)
from optuna.pruners import HyperbandPruner
import evaluate

# Loading the dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")

# Selecting 50k,5k selection for train and validation for quicker compute
train_data = dataset["train"].select(range(50000))
val_data = dataset["validation"].select(range(5000))

# Loading in BART and autotokenizer
model_name = "facebook/bart-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenization and preprocessing
def preprocess(example):
    #Tokenize the articles
    inputs = tokenizer(example["article"], max_length=1024, padding="max_length", truncation=True)

    #Tokenize the summarys
    with tokenizer.as_target_tokenizer():
        targets = tokenizer(example["highlights"], max_length=128, padding="max_length", truncation=True)

    #Set the labels for training
    inputs["labels"] = targets["input_ids"]
    return inputs

# Apply preprocessing to both train and validation datasets
train_dataset = train_data.map(preprocess, batched=True, remove_columns=train_data.column_names)
val_dataset = val_data.map(preprocess, batched=True, remove_columns=val_data.column_names)

# Define the data collator
data_collator = DataCollatorForSeq2Seq(tokenizer)

# Load ROUGE metric for evaluation
rouge = evaluate.load("rouge")

# Compute evaluation metrics during training
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    return rouge.compute(predictions=decoded_preds, references=decoded_labels)

# Model initialization function
def model_init():
    return AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Path to track best model score
best_model_tracker_path = "./bart-cnn-best-so-far/best_score.txt"
best_score = -1.0 #starting initialization of -1

# Loading last best score
if os.path.exists(best_model_tracker_path):
    with open(best_model_tracker_path, "r") as f:
        best_score = float(f.read().strip())

# Objective function for Optuna trials
def run_trial(trial):
    global best_score

    # Unique output path for each trial
    trial_output_dir = f"./bart-cnn-optuna/trial-{trial.number}"

    #training args 
    training_args = Seq2SeqTrainingArguments(
        output_dir=trial_output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"{trial_output_dir}/logs",
        predict_with_generate=True,
        fp16=torch.cuda.is_available(), 
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        greater_is_better=True,
        save_total_limit=1,
        # hyperparameters list
        per_device_train_batch_size=trial.suggest_categorical("per_device_train_batch_size", [4, 8]),
        num_train_epochs=trial.suggest_int("num_train_epochs", 2, 3),
        learning_rate=trial.suggest_categorical("learning_rate", [1e-5, 3e-5, 5e-5]),
        weight_decay=trial.suggest_categorical("weight_decay", [0.0, 0.2, 0.4])
    )

    # Instantiate Trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        model_init=model_init,
    )

    # Train model
    trainer.train()

    # Evaluate trained model
    metrics = trainer.evaluate()
    rougeL = metrics["eval_rougeL"]

    # If this trial is the best so far, save the model and tokenizer
    if rougeL > best_score:
        best_score = rougeL
        os.makedirs("./bart-cnn-best-so-far", exist_ok=True)
        trainer.save_model("./bart-cnn-best-so-far")
        tokenizer.save_pretrained("./bart-cnn-best-so-far")
        with open(best_model_tracker_path, "w") as f:
            f.write(str(best_score))
        print(f"New best model saved with ROUGE-L: {rougeL:.4f}")

    return rougeL

# Run Optuna hyperparameter optimization using Hyperband pruning
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize", pruner=HyperbandPruner())
    study.optimize(run_trial, n_trials=12)
    print("Best trial:", study.best_trial)

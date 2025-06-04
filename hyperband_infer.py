import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Loading the best model and tokenizer
model_path = "./bart-cnn-best-so-far"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

# Set model to evaluation mode
model.eval()

# GPU selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loading first 10 samples from the validation set
dataset = load_dataset("cnn_dailymail", "3.0.0", split="validation[:10]")

# Loop through the dataset and generate summaries
for idx, item in enumerate(dataset):
    article = item["article"]
    reference_summary = item["highlights"]

    # Tokenize the input article
    inputs = tokenizer("summarize: " + article, return_tensors="pt", max_length=1024, truncation=True).to(device)

    # Generating the summary using beam search
    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=128,         
            num_beams=4,            
            early_stopping=True    
        )

    # Decode the generated summary into text
    generated_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Print the start of the original article, the generated summary, and the reference summary
    print(f"\n Article {idx + 1}:\n{article[:400]}...\n")
    print(f"Generated Summary:\n{generated_summary}\n")
    print(f"Reference Summary:\n{reference_summary}\n")
    print("=" * 100)

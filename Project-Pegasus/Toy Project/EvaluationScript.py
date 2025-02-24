import json
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from bert_score import score as bert_score

# Load the fine-tuned Pegasus model and tokenizer
model_path = "pegasus-finetuned"  # Path to your fine-tuned model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to("cuda" if torch.cuda.is_available() else "cpu")

# Load evaluation dataset
eval_file = "QG_summarized_eval.json"

with open(eval_file, "r") as f:
    data = json.load(f)

# Extract contexts and reference summaries
contexts = [item["context"] for item in data["results"]]
reference_summaries = [item["summary"] for item in data["results"]]

# Generate summaries using the fine-tuned model
generated_summaries = []

def generate_summary(text, max_length=100, min_length=20, do_sample=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="longest", max_length=512).to(device)

    
    summary_ids = model.generate(
        inputs["input_ids"], 
        max_length=max_length, 
        min_length=min_length, 
        num_beams=5,  # Beam search for better summaries
        do_sample=do_sample
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Run inference on all evaluation examples
print("\nGenerating summaries for evaluation set...\n")
for context in tqdm(contexts, desc="Generating Summaries"):
    generated_summaries.append(generate_summary(context))

# Compute BERTScore
print("\nCalculating BERTScore...\n")
P, R, F1 = bert_score(generated_summaries, reference_summaries, lang="en", model_type="microsoft/deberta-xlarge-mnli")

# Compute and print average scores
avg_precision = P.mean().item()
avg_recall = R.mean().item()
avg_f1 = F1.mean().item()

print(f"\n🔹 **BERTScore Evaluation Results** 🔹")
print(f"Precision: {avg_precision:.4f}")
print(f"Recall: {avg_recall:.4f}")
print(f"F1 Score: {avg_f1:.4f}")

# Save results
results = {
    "BERTScore": {
        "Precision": avg_precision,
        "Recall": avg_recall,
        "F1": avg_f1
    }
}

with open("evaluation_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("\n✅ Evaluation Complete! Results saved to `evaluation_results.json`")

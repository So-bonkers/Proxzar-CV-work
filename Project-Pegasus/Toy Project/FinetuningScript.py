import os
import json
import logging
import torch
import wandb
from tqdm import tqdm
from datasets import Dataset
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, 
                          DataCollatorForSeq2Seq, TrainingArguments, Trainer)

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

WANDB_API_KEY = "cdfdf1645663ac737d50aa31937f8d4c6de55ade"  

# Login to W&B
wandb.login(key=WANDB_API_KEY)
wandb.init(project="pegasus-summarization", name="pegasus-finetune")

# Device check
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# Load model & tokenizer
try:
    model_ckpt = "google/pegasus-x-base"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt).to(device)
    logger.info("Successfully loaded Pegasus model and tokenizer.")
except Exception as e:
    logger.error(f"Error loading model/tokenizer: {e}")
    raise

# Load dataset
logger.info("Loading dataset...")
def load_dataset_from_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    contexts = [item["context"] for item in data["results"]]
    summaries = [item["summary"] for item in data["results"]]
    return Dataset.from_dict({"context": contexts, "summary": summaries})

dataset = load_dataset_from_json("QG_summarized_range_1.json")

# Split dataset: 80% train, 10% validation, 10% test
logger.info("Splitting dataset...")
dataset = dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)
temp_dataset = dataset["test"].train_test_split(test_size=0.5, shuffle=True, seed=42)
train_dataset, val_dataset, test_dataset = dataset["train"], temp_dataset["train"], temp_dataset["test"]
logger.info(f"Dataset sizes - Train: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}")

# Preprocessing function
def preprocess_function(example_batch):
    encodings = tokenizer(example_batch["context"], text_target=example_batch["summary"],
                          max_length=1024, truncation=True)
    return {'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': encodings['labels']}

# Tokenization with tqdm progress bar
logger.info("Tokenizing datasets...")
train_dataset = train_dataset.map(preprocess_function, batched=True, desc="Tokenizing Train")
val_dataset = val_dataset.map(preprocess_function, batched=True, desc="Tokenizing Validation")
test_dataset = test_dataset.map(preprocess_function, batched=True, desc="Tokenizing Test")

# Convert datasets to PyTorch format
columns = ['input_ids', 'labels', 'attention_mask']
train_dataset.set_format(type='torch', columns=columns)
val_dataset.set_format(type='torch', columns=columns)
test_dataset.set_format(type='torch', columns=columns)

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)

# Training arguments with batch size
BATCH_SIZE = 32  

training_args = TrainingArguments(
    output_dir="pegasus-custom",
    num_train_epochs=10,
    warmup_steps=500,
    per_device_train_batch_size=BATCH_SIZE,  # Train batch size
    per_device_eval_batch_size=BATCH_SIZE,  # Validation/Test batch size
    weight_decay=0.01,
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=1000,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    report_to="wandb"
)

# Trainer
trainer = Trainer(
    model=model_pegasus,
    args=training_args,
    tokenizer=tokenizer,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Start training
logger.info("Starting fine-tuning...")
try:
    trainer.train()
    logger.info("Training completed successfully!")
except Exception as e:
    logger.error(f"Error during training: {e}")
    raise

# Save model
trainer.save_model("pegasus-finetuned")
logger.info("Model saved as 'pegasus-finetuned'.")

# Evaluate on test set
logger.info("Evaluating model on test set...")
metrics = trainer.evaluate(test_dataset)
logger.info(f"Test set evaluation metrics: {metrics}")

# Log test results to W&B
wandb.log(metrics)

# Finish W&B run
wandb.finish()
logger.info("W&B run completed successfully!")
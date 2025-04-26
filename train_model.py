import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import Dataset
import os
from tqdm import tqdm
import logging
import re
from datasets import load_dataset

# Set logging level
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_wikipedia_dataset():
    """Load Wikipedia dataset"""
    print("Loading Wikipedia dataset...")
    dataset = load_dataset("wikipedia", "20220301.en", split="train[:1%]")
    print(f"Dataset loaded successfully, size: {len(dataset)}")
    return dataset

def preprocess_dataset(dataset):
    """Preprocess dataset"""
    print("Preprocessing dataset...")
    
    def process_example(example):
        # Remove special characters and extra spaces
        text = re.sub(r'\s+', ' ', example['text'])
        text = re.sub(r'[^\w\s]', '', text)
        return {'text': text}
    
    processed_dataset = dataset.map(process_example, batched=False)
    print("Dataset preprocessing completed")
    return processed_dataset

def train_model(dataset, model_name="facebook/bart-large-cnn", output_dir="./models/wiki_summarizer"):
    """Train the model"""
    print("Starting model training...")
    
    # Load tokenizer and model
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    
    # Prepare training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=1000,
        save_total_limit=2,
        logging_steps=100,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=500,
        fp16=True,
        gradient_accumulation_steps=4,
    )
    
    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    
    # Start training
    print("Training started...")
    trainer.train()
    print("Training completed!")
    
    # Save model
    print(f"Saving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Model saved successfully!")

def main():
    """Main function"""
    print("Starting training process...")
    
    # Load dataset
    dataset = load_wikipedia_dataset()
    
    # Preprocess dataset
    processed_dataset = preprocess_dataset(dataset)
    
    # Train model
    train_model(processed_dataset)
    
    print("Training process completed successfully!")

if __name__ == "__main__":
    main()

def load_wiki_data(filepath="data/wiki_training_data.json"):
    """
    Load Wikipedia data
    """
    logger.info("Loading training data...")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Prepare training data
    train_data = {
        "text": [article["full_content"] for article in data],
        "summary": [article["summary"] for article in data]
    }
    
    logger.info(f"Loaded {len(train_data['text'])} training samples")
    return Dataset.from_dict(train_data)

def train_model(data, model_name="facebook/bart-large-cnn", output_dir="models/wiki_summarizer"):
    """
    Train the summarization model
    """
    logger.info("Initializing model and tokenizer...")
    # Load pretrained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Data preprocessing function
    def preprocess_function(examples):
        inputs = examples["text"]
        targets = examples["summary"]
        
        # Encode inputs and outputs
        model_inputs = tokenizer(inputs, max_length=1024, truncation=True, padding="max_length")
        labels = tokenizer(targets, max_length=256, truncation=True, padding="max_length")
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    logger.info("Preprocessing data...")
    # Preprocess data
    tokenized_data = data.map(preprocess_function, batched=True)
    
    # Set training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=3,
        predict_with_generate=True,
        fp16=True,
        push_to_hub=False,
        logging_steps=100,
        logging_dir="logs",
        report_to="tensorboard",
    )
    
    # Create trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data,
        tokenizer=tokenizer,
    )
    
    # Start training
    logger.info("Starting model training...")
    trainer.train()
    
    # Save model
    trainer.save_model(output_dir)
    logger.info(f"Model saved to: {output_dir}")

if __name__ == "__main__":
    try:
        # Check if GPU is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        if device.type == 'cuda':
            logger.info(f"GPU Model: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Load data
        train_data = load_wiki_data()
        
        # Train model
        train_model(train_data)
        
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
    except Exception as e:
        logger.error(f"Error during training: {str(e)}") 
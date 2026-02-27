"""
Training Script for TahananSafe AI
Fine-tunes Qwen/Qwen2.5-0.5B-Instruct using supervised fine-tuning with LoRA.
"""

import os

# Force Transformers to use PyTorch-only (no TensorFlow/Keras imports)
os.environ["TRANSFORMERS_NO_TF"] = "1"

import yaml
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_from_disk


class IncidentReportTrainer:
    """Trainer for fine-tuning Qwen/Qwen2.5-0.5B-Instruct on incident reports."""
    
    def __init__(self, config_path: str = "training/config.yaml"):
        """Initialize trainer with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['model']
        self.lora_config = self.config['lora']
        self.training_config = self.config['training']
        
    def load_model_and_tokenizer(self):
        """Load base model and tokenizer"""
        print(f"Loading model: {self.model_config['base_model']}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config['base_model'],
            trust_remote_code=True
        )
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with quantization if specified
        if self.model_config.get('use_4bit', False):
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_config['base_model'],
                quantization_config=quantization_config,
                device_map=self.model_config.get('device_map', 'auto'),
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
        elif self.model_config.get('use_8bit', False):
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_config['base_model'],
                load_in_8bit=True,
                device_map=self.model_config.get('device_map', 'auto'),
                trust_remote_code=True
            )
        else:
            # Default path: let Transformers/Accelerate place the model on GPU if available.
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_config['base_model'],
                device_map=self.model_config.get("device_map", "auto"),
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
        
        # Prepare model for k-bit training
        if self.model_config.get('use_4bit', False) or self.model_config.get('use_8bit', False):
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=self.lora_config['r'],
            lora_alpha=self.lora_config['lora_alpha'],
            target_modules=self.lora_config['target_modules'],
            lora_dropout=self.lora_config['lora_dropout'],
            bias=self.lora_config['bias'],
            task_type=self.lora_config['task_type']
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        print("Model and tokenizer loaded successfully!")
    
    def load_datasets(self):
        """Load processed datasets"""
        dataset_path = self.config['dataset']['processed_path']
        print(f"Loading datasets from {dataset_path}...")
        
        self.datasets = load_from_disk(dataset_path)
        print(f"Train examples: {len(self.datasets['train'])}")
        print(f"Validation examples: {len(self.datasets['validation'])}")
        print(f"Test examples: {len(self.datasets['test'])}")
    
    def tokenize_function(self, examples):
        """Tokenize dataset examples"""
        # Tokenize the 'text' field
        tokenized = self.tokenizer(
            examples['text'],
            truncation=True,
            max_length=self.config['dataset']['max_length'],
            padding='max_length'
        )
        
        # Create labels (same as input_ids for causal LM)
        tokenized['labels'] = tokenized['input_ids'].copy()
        
        return tokenized
    
    def train(self):
        """Main training function"""
        # Load model and tokenizer
        self.load_model_and_tokenizer()
        
        # Load datasets
        self.load_datasets()
        
        # Tokenize datasets
        print("Tokenizing datasets...")
        tokenized_train = self.datasets['train'].map(
            self.tokenize_function,
            batched=True,
            remove_columns=self.datasets['train'].column_names
        )
        
        tokenized_val = self.datasets['validation'].map(
            self.tokenize_function,
            batched=True,
            remove_columns=self.datasets['validation'].column_names
        )
        
        # Training arguments
        # Note: some older versions of `transformers` don't support
        # `evaluation_strategy`, so we keep args minimal and call
        # `trainer.evaluate` manually after training.
        training_args = TrainingArguments(
            output_dir=self.training_config['output_dir'],
            num_train_epochs=self.training_config['num_train_epochs'],
            per_device_train_batch_size=self.training_config['per_device_train_batch_size'],
            gradient_accumulation_steps=self.training_config['gradient_accumulation_steps'],
            # Ensure numeric types for HF Trainer / torch optimizer
            learning_rate=float(self.training_config['learning_rate']),
            warmup_steps=self.training_config['warmup_steps'],
            logging_steps=self.training_config['logging_steps'],
            save_steps=self.training_config['save_steps'],
            save_total_limit=self.training_config['save_total_limit'],
            fp16=self.training_config.get('fp16', True),
            optim=self.training_config.get('optim', 'adamw_torch'),
            lr_scheduler_type=self.training_config['lr_scheduler_type']
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Causal LM, not masked LM
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            data_collator=data_collator,
        )
        
        # Train
        print("Starting training...")
        trainer.train()
        
        # Save final model
        print(f"Saving model to {self.training_config['output_dir']}...")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.training_config['output_dir'])
        
        print("Training complete!")
        
        # Evaluate on test set
        print("Evaluating on test set...")
        tokenized_test = self.datasets['test'].map(
            self.tokenize_function,
            batched=True,
            remove_columns=self.datasets['test'].column_names
        )
        
        test_results = trainer.evaluate(eval_dataset=tokenized_test)
        print(f"Test results: {test_results}")


if __name__ == "__main__":
    trainer = IncidentReportTrainer()
    trainer.train()

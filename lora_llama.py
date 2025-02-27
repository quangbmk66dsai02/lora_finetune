from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

# Load the pre-trained model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
)

from datasets import load_dataset

dataset  = load_dataset("gbharti/finance-alpaca")

# Tokenize the dataset
def tokenize_function(examples):
    # Combine instruction and input for the prompt
    prompts = [f"Instruction: {instr}\nInput: {inp}\nResponse:" for instr, inp in zip(examples['instruction'], examples['input'])]
    # Tokenize the prompts and outputs
    model_inputs = tokenizer(prompts, padding="max_length", truncation=True, max_length=512)
    labels = tokenizer(examples['output'], padding="max_length", truncation=True, max_length=512)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Tokenize the dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)
print("TOKENIZED DATASET", tokenized_dataset)

from peft import LoraConfig, get_peft_model

# Define LoRA configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

from transformers import TrainingArguments, Trainer

# Define training arguments
training_args = TrainingArguments(
    output_dir="./lora-llama3.2-finance",
    evaluation_strategy="no",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    fp16=True  # Enable mixed precision training
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
 
)

# Start training
trainer.train()
model.save_pretrained("./lora-llama3.2-finance")

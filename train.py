from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset

MODEL = "google/flan-t5-small"
DATASET = "Helsinki-NLP/opus-100"

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

from datasets import load_dataset
dataset = load_dataset(DATASET, "en-ne")

instruction = "Translate English to Nepali: "

def tokenize_function(examples):
    inputs = [instruction + example["en"] for example in examples["translation"]]
    targets = [example["ne"] for example in examples["translation"]]
    return tokenizer(inputs, text_target=targets, max_length=128, truncation=True)

dataset = dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

training_args=Seq2SeqTrainingArguments(
    output_dir="",
    evaluation_strategy="epoch",
    learning_rate=2e-05,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    save_total_limit=3,
    predict_with_generate=True,
    lr_scheduler_type="linear",
    num_train_epochs=1,
    #push_to_hub=True
)

trainer=Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

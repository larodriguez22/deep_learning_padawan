# Login to Hugging Face
from huggingface_hub import login
login()

# Login to Weights and Biases
import wandb
wandb.login()

source_lang, source_lang_iso = "spa", "spa"
target_lang, target_lang_iso = "guc", "guc" # or pbb, Paez
base_model = "t5-base" # or t5-small, t5-large, google/mt5-base, facebook/bart-large, etc

from datasets import load_dataset

dataset = load_dataset(f"lrodriguez22/translation_{source_lang_iso}_{target_lang_iso}")

from transformers import AutoTokenizer

checkpoint = base_model
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

prefix = f"translate {source_lang} to {target_lang}: "

def preprocess_function(examples):
    inputs = [prefix + example[source_lang] for example in examples["translation"]]
    targets = [example[target_lang] for example in examples["translation"]]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=512, truncation=True)
    return model_inputs

print(dataset["train"][0])

tokenized_dataset = dataset.map(preprocess_function, batched=True)

from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)

import numpy as np
import evaluate

metric = evaluate.load("sacrebleu")

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

new_model_name = f'{base_model}-translation-{source_lang_iso}-{target_lang_iso}'

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

from transformers import EarlyStoppingCallback

training_args = Seq2SeqTrainingArguments(
    output_dir=f"./results/{new_model_name}",
    eval_strategy="epoch",
    save_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    fp16=False,
    optim="adafactor",
    weight_decay=0.01,
    save_total_limit=5,
    num_train_epochs=10,
    predict_with_generate=True,
    push_to_hub=True,
    load_best_model_at_end = True,
    report_to="wandb",
    warmup_steps=10,
    logging_steps=1,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
)

trainer.train()

trainer.push_to_hub()



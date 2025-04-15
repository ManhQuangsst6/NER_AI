# pip install datasets transformers evaluate seqeval torch pandas pyarrow

import json

# Load the JSON data from the file
with open('dataset1.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Initialize the Hugging Face dataset format
hf_dataset = []

# Define a function to tokenize text into words
def tokenize_text(text):
    return text.split()

# Iterate through each item in the JSON data
for item in data:
    # Extract the necessary information
    text = item['data']['text']
    annotations = item['annotations'][0]['result']

    # Initialize tokens and ner_tags
    tokens = tokenize_text(text)
    ner_tags = [0] * len(tokens)

    # Assign NER tags to tokens
    for annotation in annotations:
        start = annotation['value']['start']
        end = annotation['value']['end']
        label = annotation['value']['labels'][0]

        # Find the token indices for the annotation
        token_start_idx = len(tokenize_text(text[:start]))
        token_end_idx = len(tokenize_text(text[:end]))

        # Assign the label to the corresponding tokens
        for i in range(token_start_idx, token_end_idx):
            ner_tags[i] = label

    # Create a dictionary for the Hugging Face dataset format
    hf_item = {
        "id": str(item['id']),
        "ner_tags": ner_tags,
        "tokens": tokens
    }

    # Append the item to the Hugging Face dataset
    hf_dataset.append(hf_item)

# Save the Hugging Face dataset to a new JSON file
with open('hf_dataset.json', 'w', encoding='utf-8') as file:
    json.dump(hf_dataset, file, ensure_ascii=False, indent=4)

# Load the JSON data from the file
with open('hf_dataset.json', 'r', encoding='utf-8') as file:
    hf_dataset = json.load(file)

# Function to add B- and I- tags to ner_tags
def add_bio_tags(tokens, ner_tags):
    bio_tags = []
    prev_tag = 'O'
    for i, tag in enumerate(ner_tags):
        if tag == 0:
            bio_tags.append(0)
            prev_tag = 'O'
        else:
            if prev_tag == 'O' or prev_tag != tag:
                bio_tags.append(f'B-{tag}')
            else:
                bio_tags.append(f'I-{tag}')
            prev_tag = tag
    return bio_tags

# Update the ner_tags in the dataset
for item in hf_dataset:
    item['ner_tags'] = add_bio_tags(item['tokens'], item['ner_tags'])

# Save the updated dataset to a new JSON file
with open('hf_dataset_bio.json', 'w', encoding='utf-8') as file:
    json.dump(hf_dataset, file, ensure_ascii=False, indent=4)

import pandas as pd
import itertools

# Load the JSON file into a pandas DataFrame
df = pd.read_json('hf_dataset_bio.json')

labels = [
    "Name", "Sex", "PhoneNumber", "Email", "Address"
]
label_list = ["O"]
for lbl in labels:
    label_list.append(f'B-{lbl}')
    label_list.append(f'I-{lbl}')

import numpy as np
import evaluate

import json
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification

id2label = {}
for i, label in enumerate(label_list):
    id2label[i] = label

label2id = {}
for i, label in enumerate(label_list):
    label2id[label] = i

seqeval = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(
        predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# Apply the mapping to the 'ner_tags' column
df['ner_tags'] = df['ner_tags'].apply(lambda x: [label2id.get(i, i) for i in x])

# Convert the pandas DataFrame to a Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Load the tokenizer and model
model_name = "google-bert/bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="caches/")
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(label_list), id2label=id2label, label2id=label2id, cache_dir="caches/")

# Tokenize the dataset
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples['tokens'], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples['ner_tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

label_all_tokens = True
tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

# Create data collator
data_collator = DataCollatorForTokenClassification(tokenizer)

# Set training arguments
training_args = TrainingArguments(
    output_dir="./results",
    # evaluation_strategy='steps',
    eval_steps=500,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=1,
    num_train_epochs=10,
    # load_best_model_at_end=True,
    report_to="none",
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

model.save_pretrained('mr4/YYY')
tokenizer.save_pretrained('mr4/YYY')

text = "Xin chào, tôi tên là Đặng Việt Dũng, tôi sinh ra ở Hòa Bình"

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

model_name = "mr4/YYY"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_token_class_ids = torch.argmax(logits, dim=-1)
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())
predicted_entities = [model.config.id2label[id] for id in predicted_token_class_ids.squeeze().tolist()]

final_result = []
current_entity = []
current_label = None

for token, entity in zip(tokens, predicted_entities):
    if entity.startswith("B-"):
        if entity[2:] == current_label:
            current_entity.append(token)
        else:
            if current_entity:
                final_result.append((" ".join(current_entity), current_label))
            current_entity = [token]
            current_label = entity[2:]
    elif entity.startswith("I-") and current_label == entity[2:]:
        current_entity.append(token)
    else:
        if current_entity:
            final_result.append((" ".join(current_entity), current_label))
        current_entity = []
        current_label = None
if current_entity:
    final_result.append((" ".join(current_entity), current_label))

for entity, label in final_result:
    ner_value = entity.replace(" ##", "").replace("##", "")
    print(f"  - {ner_value}: {label}")

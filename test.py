text = "Hello! Tôi là Thanh Duy, một chàng trai đam mê công nghệ, sống tại Hà Nội. Bạn có thể liên lạc với tôi qua email thanhduy95@gmail.com hoặc gọi vào số 0902 345 678. Mong sẽ có cơ hội chia sẻ và học hỏi cùng các bạn!"

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

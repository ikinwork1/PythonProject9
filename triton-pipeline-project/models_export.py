import os
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTModelForFeatureExtraction, ORTModelForCausalLM

def create_config(path, content):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "config.pbtxt"), "w", encoding="utf-8") as f:
        f.write(content.strip())

# 1. Пути
base_repo = "./model_repository"

# 2. Скачивание и конвертация в ONNX
print("Экспорт BERT...")
bert_model_path = os.path.join(base_repo, "bert_classifier", "1")
model_bert = ORTModelForSequenceClassification.from_pretrained("fabriceyhc/bert-base-uncased-ag_news", export=True)
model_bert.save_pretrained(bert_model_path)
os.rename(os.path.join(bert_model_path, "model.onnx"), os.path.join(bert_model_path, "model.onnx")) # Убедимся в имени

print("Экспорт Sentence-Transformer...")
st_model_path = os.path.join(base_repo, "sentence_transformer", "1")
model_st = ORTModelForFeatureExtraction.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", export=True)
model_st.save_pretrained(st_model_path)

print("Экспорт Qwen-0.5B...")
qwen_model_path = os.path.join(base_repo, "qwen_generation", "1")
model_qwen = ORTModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", export=True)
model_qwen.save_pretrained(qwen_model_path)

# 3. Запись конфигураций Triton
create_config("./model_repository/bert_classifier", """
name: "bert_classifier"
platform: "onnxruntime_onnx"
max_batch_size: 32
input [
  { name: "input_ids"; data_type: TYPE_INT64; dims: [ -1 ] },
  { name: "attention_mask"; data_type: TYPE_INT64; dims: [ -1 ] }
]
output [ { name: "logits"; data_type: TYPE_FP32; dims: [ 4 ] } ]
dynamic_batching { }
""")

create_config("./model_repository/sentence_transformer", """
name: "sentence_transformer"
platform: "onnxruntime_onnx"
max_batch_size: 32
input [
  { name: "input_ids"; data_type: TYPE_INT64; dims: [ -1 ] },
  { name: "attention_mask"; data_type: TYPE_INT64; dims: [ -1 ] }
]
output [ { name: "last_hidden_state"; data_type: TYPE_FP32; dims: [ -1, 384 ] } ]
dynamic_batching { }
""")

create_config("./model_repository/qwen_generation", """
name: "qwen_generation"
platform: "onnxruntime_onnx"
max_batch_size: 16
input [
  { name: "input_ids"; data_type: TYPE_INT64; dims: [ -1 ] },
  { name: "attention_mask"; data_type: TYPE_INT64; dims: [ -1 ] }
]
output [ { name: "logits"; data_type: TYPE_FP32; dims: [ -1, 151936 ] } ]
dynamic_batching { }
""")

print("Все модели подготовлены локально!")
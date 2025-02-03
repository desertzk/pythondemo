from transformers import AutoModel, AutoTokenizer
import torch


# 下载并加载模型和分词器
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 准备输入文本
texts = ["Hello, how are you?", "Nice to meet you!"]

# 对输入文本进行编码
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)

# 模型推断
with torch.no_grad():
    outputs = model(**inputs)

# 输出最后一层隐藏状态的结果
last_hidden_states = outputs.last_hidden_state

print("Last hidden states shape:", last_hidden_states.size())

# 如果需要进一步处理输出，比如提取[CLS]标记的表示作为句子向量
cls_representation = last_hidden_states[:, 0, :].numpy()  # 提取[CLS]标记的表示
print("CLS token representations shape:", cls_representation.shape)
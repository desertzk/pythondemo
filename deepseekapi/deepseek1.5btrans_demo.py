from modelscope import AutoModelForCausalLM, AutoTokenizer

# 输入模型下载地址
model_name = "./DeepSeek-R1-1.5B"

# 实例化预训练模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    low_cpu_mem_usage=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# 创建消息
prompt = "你好，好久不见，请介绍下你自己。"
messages = [
    {"role": "system", "content": "你是一名助人为乐的助手。"},
    {"role": "user", "content": prompt}
]


# 分词
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# 创建回复
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(response)

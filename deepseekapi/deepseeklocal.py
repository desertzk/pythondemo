from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 下载并加载模型和分词器
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 创建对话历史存储
conversation_history = []

# 定义系统提示（可选）
system_prompt = "You are a helpful assistant."

while True:
    # 获取用户输入
    user_input = input("\nUser: ")

    # 退出条件
    if user_input.lower() in ["exit", "quit"]:
        break

    # 将用户输入加入对话历史
    conversation_history.append({"role": "user", "content": user_input})

    # 使用 apply_chat_template() 生成模型输入，确保返回 dict 包含 input_ids 和 attention_mask
    inputs = tokenizer.apply_chat_template(
        conversation=conversation_history,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    )

    # 将所有张量移动到模型所在的设备上
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # 生成回复，同时传入 attention_mask
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=512,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

    # 解码响应并移除输入部分的 tokens
    response = tokenizer.decode(
        outputs[0][len(inputs["input_ids"][0]):],
        skip_special_tokens=True
    ).strip()

    # 将助手回复加入对话历史
    conversation_history.append({"role": "assistant", "content": response})

    # 打印助手回复
    print(f"\nAssistant: {response}")

# 退出提示
print("\nChat session ended.")

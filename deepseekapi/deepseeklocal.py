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

# 定义系统提示
system_prompt = "You are a helpful assistant."

while True:
    # 获取用户输入
    user_input = input("\nUser: ")
    
    # 退出条件
    if user_input.lower() in ["exit", "quit"]:
        break
    
    # 将用户输入加入对话历史
    conversation_history.append({"role": "user", "content": user_input})
    
    # 生成模型输入
    input_ids = tokenizer.apply_chat_template(
        conversation=conversation_history,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    # 生成回复
    outputs = model.generate(
        input_ids,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # 解码响应并移除特殊标记
    response = tokenizer.decode(
        outputs[0][len(input_ids[0]):],
        skip_special_tokens=True
    ).strip()
    
    # 将助手回复加入对话历史
    conversation_history.append({"role": "assistant", "content": response})
    
    # 打印助手回复
    print(f"\nAssistant: {response}")

# 退出提示
print("\nChat session ended.")

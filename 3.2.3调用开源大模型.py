import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import platform

# 指定模型ID
model_id = "Qwen/Qwen1.5-0.5B-Chat"

# 设置设备，优先使用GPU
device = "mps" if platform.system() == "Darwin" else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 加载模型，并将其移动到指定设备
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

print("模型和分词器加载完成！")

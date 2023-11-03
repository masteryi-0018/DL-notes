'''1. 模型加载'''
# chatglm2
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("./chatglm2-6b-int4", trust_remote_code=True)
model = AutoModel.from_pretrained("./chatglm2-6b-int4", trust_remote_code=True).float()
model = model.eval()

response, history = model.chat(tokenizer, "你好", history=[])
print(response)

# llama2
# 存在bug：cpu版本torch依赖flash_attn，而安装flash_attn需要nvcc，报错位置为modeling_flash_llama.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("./LLaMA-2-7B-32K", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("./LLaMA-2-7B-32K", trust_remote_code=True, torch_dtype=torch.float16)

input_context = "hello"
input_ids = tokenizer.encode(input_context, return_tensors="pt")
output = model.generate(input_ids, max_length=128, temperature=0.7)
output_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(output_text)
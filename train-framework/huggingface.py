'''1. 模型加载'''
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("./chatglm2-6b-int4", trust_remote_code=True)
model = AutoModel.from_pretrained("./chatglm2-6b-int4", trust_remote_code=True).float()
model = model.eval()

response, history = model.chat(tokenizer, "你好", history=[])
print(response)
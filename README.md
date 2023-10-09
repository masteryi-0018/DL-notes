# llm-notes

## 模型获取

目前开源的常用的模型：
- chatglm
- baichuan

更全面的开源llm整理：<https://zhuanlan.zhihu.com/p/654956859>

### chatglm

网络情况“良好”的状态下，直接利用hugging face官方下载，示例脚本：
```py
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm2-6b-int4", trust_remote_code=True).float()
model = model.eval()
response, history = model.chat(tokenizer, "你好", history=[])
print(response)
```

可以看到代码从hf.co下载了需要的模型，鉴于有很多人网络“不好”，以及最近hf.co似乎被墙了，需要一种更加稳妥的方式来学习llm。参考回答：<https://www.zhihu.com/question/599683557/answer/3202372678>

幸好清华将模型权重上传了一份到自己的服务器，链接：[chatglm](https://cloud.tsinghua.edu.cn/d/674208019e314311ab5c/)

但是只有这些模型是不够的，还需要一些其他的文件，从而顺利加载模型，根据自动下载的文件可知：

加载tokenizer需要：
- tokenizer_config.json
- tokenization_chatglm.py
- tokenizer.model（已有）

加载model需要：
- config.json
- configuration_chatglm.py
- modeling_chatglm.py
- quantization.py（可选，运行int4等量化模型需要）
- pytorch_model.bin（已有）

于是从清华云下载权重后，再将此仓库的其他文件一并放入文件夹，最后将上述脚本的文件修改为本地文件夹路径例如`"./chatglm2-6b-int4"`，，即可加载运行

## HF模型转换为onnx模型
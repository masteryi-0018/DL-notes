# llm

## 模型获取

目前开源的常用的模型：
- chatglm
- baichuan
- llama

更全面的开源llm整理：<https://zhuanlan.zhihu.com/p/654956859>

## chatglm

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

## llama

官方的模型需要申请license才能下载，好在有其他方法可以尝试，以下链接：

1. hugging face非官方
- <https://huggingface.co/NousResearch/Llama-2-13b-hf>
- <https://huggingface.co/NousResearch/Llama-2-7b-hf>
- <https://huggingface.co/NousResearch/Llama-2-7b-chat-hf>

2. gitee镜像仓库
- <https://gitee.com/modelee/LLaMA-2-7B-32K>
- <https://gitee.com/modelee/llama-7b-hf>
- <https://gitee.com/modelee/llama-7b>

3. 异型岛
- <https://aliendao.cn/models/NousResearch/Llama-2-7b-hf>

我的尝试

```
git clone https://huggingface.co/NousResearch/Llama-2-7b-chat-hf
git restore --source=HEAD :/
```

## HF模型转换为onnx模型

参考链接：[大模型部署：huggingface模型转onnx格式](https://zhuanlan.zhihu.com/p/660330173)
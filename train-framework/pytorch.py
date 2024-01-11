'''1. 下载模型'''

# 方法1
import torch
model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
model.eval()
print(model)

# 方法2
from torchvision.models import mobilenet_v2
model = mobilenet_v2(weights='IMAGENET1K_V2')
model.eval()
print(model)


'''2. 读取模型'''

import torch
from torchvision.models import mobilenet_v2
from torchinfo import summary

# 自己的默认路径
path = "C:/Users/xxx/.cache/torch/hub/checkpoints/mobilenet_v2-7ebf99e0.pth"
ckpt = torch.load(path)

model = mobilenet_v2(weights=None)
model.load_state_dict(ckpt)
model.eval()

batch_size = 1
summary(model, input_size=(batch_size, 3, 28, 28))# , verbose = 2)


'''3. 保存模型'''

import torch
# 保存整个模型
save_path = "mobilenetv2.pth"
torch.save(model, save_path)

# 加载整个模型
model = torch.load(save_path)


'''4. 转换模型'''

import torch
# Input to the model
input = torch.ones(1, 3, 224, 224)

# Export the model
torch.onnx.export(model,                     # model being run
                  input,                     # model input (or a tuple for multiple inputs)
                  "mobilenetv2_pt.onnx",        # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=15,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                #   dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                #                 'output' : {0 : 'batch_size'}}
                  )

'''5. 量化'''

# a. 下载预训练的量化模型
from torchvision.models.quantization import mobilenet_v2
model_quantized = mobilenet_v2(weights='MobileNet_V2_QuantizedWeights.IMAGENET1K_QNNPACK_V1', quantize=True)  # TODO
torch.save(model_quantized.state_dict(), "model_quantized.pt")


# b. PTQ动态量化
import torch
import torchvision
model = torchvision.models.mobilenet_v2(weights='MobileNet_V2_Weights.IMAGENET1K_V1')

model_dynamic_quantized = torch.quantization.quantize_dynamic(
    model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8
)
torch.save(model_dynamic_quantized.state_dict(), "model_dynamic_quantized.pt")

# c. PTQ静态量化
backend = "x86"
model.qconfig = torch.quantization.get_default_qconfig(backend)
torch.backends.quantized.engine = backend
model_static_quantized = torch.quantization.prepare(model, inplace=False)
model_static_quantized = torch.quantization.convert(model_static_quantized, inplace=False)
torch.save(model_static_quantized.state_dict(), "model_static_quantized.pt")

# d. QAT
# 需要有模型的代码，init插入：
# self.quant = torch.quantization.QuantStub()
# self.dequant = torch.quantization.DeQuantStub()
# forward插入：
# x = self.quant(x)
# x = self.dequant(x)

# 执行代码
model.qconfig = torch.quantization.get_default_qat_qconfig(backend)
model_qat = torch.quantization.prepare_qat(model, inplace=False)
# quantization aware training goes here
model_qat = torch.quantization.convert(model_qat.eval(), inplace=False)
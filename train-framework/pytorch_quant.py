'''1. PYTORCH 2 之前的量化'''

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


'''2. PYTORCH 2 之后的量化'''
# 需要在Linux的环境下才能完成compile

# a. PTQ
import torch
import torchvision.models as models
from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer
import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
from torch._export import capture_pre_autograd_graph
import copy

# Create the Eager Model
model_name = "resnet18"
model = models.__dict__[model_name](pretrained=True)

# Set the model to eval mode
model = model.eval()

# 这里也使用了随机数据，与tf类似，说明是静态量化
# Create the data, using the dummy data here as an example
traced_bs = 50
x = torch.randn(traced_bs, 3, 224, 224).contiguous(memory_format=torch.channels_last)
example_inputs = (x,)

# Capture the FX Graph to be quantized
with torch.no_grad():
     # if you are using the PyTorch nightlies or building from source with the pytorch master,
    # use the API of `capture_pre_autograd_graph`
    # Note 1: `capture_pre_autograd_graph` is also a short-term API, it will be updated to use the official `torch.export` API when that is ready.
    # exported_model = capture_pre_autograd_graph(
    #     model,
    #     example_inputs
    # )
    # Note 2: if you are using the PyTorch 2.1 release binary or building from source with the PyTorch 2.1 release branch,
    # please use the API of `torch._dynamo.export` to capture the FX Graph.
    exported_model, guards = torch._dynamo.export(
        model,
        *copy.deepcopy(example_inputs),
        aten_graph=True,
    )

# 配置量化器
quantizer = X86InductorQuantizer()
quantizer.set_global(xiq.get_default_x86_inductor_quantization_config())

# 准备模型
prepared_model = prepare_pt2e(exported_model, quantizer)

# We use the dummy data as an example here
prepared_model(*example_inputs)

# Alternatively: user can define the dataset to calibrate
# def calibrate(model, data_loader):
#     model.eval()
#     with torch.no_grad():
#         for image, target in data_loader:
#             model(image)
# calibrate(prepared_model, data_loader_test)  # run calibration on sample data

converted_model = convert_pt2e(prepared_model)

# 如果生成cpp的模型：
# Optional: using the C++ wrapper instead of default Python wrapper
# import torch._inductor.config as config
# config.cpp_wrapper = True

with torch.no_grad():
    optimized_model = torch.compile(converted_model)
    # Running some benchmark
    # optimized_model(*example_inputs)
    torch.save(optimized_model.state_dict(), "optimized_model.pt")

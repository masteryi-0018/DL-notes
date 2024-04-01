# 移动端推理框架

## paddle-lite

<https://github.com/PaddlePaddle/Paddle-Lite>

大致的步骤分为：

- x2paddle：转换模型。将模型（onnx、pt、tf）转换为paddle模式。默认为`uncombined`模型，即`model.pdmodel`和`model.pdiparams`等文件在一个文件夹中，使用时输入文件夹
- opt：优化模型。优化具体的行为：*量化*、子图融合、混合调度、Kernel 优选等等方法
- run：运行模型

其中，关于量化：
1. Paddle Lite OPT 工具和 PaddleSlim 都提供了动态离线量化功能，两者原理相似，都可以产出动态离线量化的模型。
2. 动态离线量化不需要任何其他的数据，静态离线量化需要少量校准数据计算量化因子（有两种计算量化因子的方法，非饱和量化方法和饱和量化方法）
3. 动态离线量化模型有两种预测方式：
    - 第一种是**反量化预测方式**，即是首先将 INT8/16 类型的权重反量化成 FP32 类型，然后再使用 FP32 浮运算运算进行预测；
    - 第二种是**量化预测方式**，即是预测中动态计算量化 OP 输入的量化信息，基于量化的输入和权重进行 INT8 整形运算。
    - 注意：目前 Paddle Lite 仅支持第一种反量化预测方式。
4. 静态离线量化可以进行量化预测，加速效果更好

优势：
- 对生态支持力度比较大，文档详细，编译简单

不足：
- 对于Tensor的功能较少，例如不能实现指定形状的Tensor生成
- 模型处理输入输出比较繁琐，没有类似pytorch的`output=model.forward()`方法简单直观

## MNN

<https://github.com/alibaba/MNN>

> 注意：Ubuntu 18.04.2 LTS下编译不通过，需要Ubuntu 20.04.3 LTS

大致的步骤分为：
- 使用MNNConverter将模型转换为mnn格式，支持的模型：TF, CAFFE, ONNX, TFLITE, TORCH, MNN, JSON
- 使用quantized.out进行量化
- 构建程序编译

关于量化：
- 离线量化：EMA，KL，ADMM
- 训练量化：LSQ，OAQ，WAQ
- **直接权值量化**：包括对称量化，非对称量化；一般8bit，计算时还原为float，这个应该和pdlite的动态离线量化一个道理
- 训练权值量化：包括对称量化；更低比特
- FP16

优势：
- 模型转换支持框架多、算子多
- 模型处理输入输出直观方便

## NCNN

<https://github.com/Tencent/ncnn>

## lite.ai.toolkit 3.2k

<https://github.com/DefTruth/lite.ai.toolkit>

好多框架与网络的集合
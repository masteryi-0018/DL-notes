# 1. 编译转换工具
```sh
mkdir build && cd build
cmake .. -DMNN_BUILD_CONVERTER=ON -DMNN_BUILD_TORCH=ON
make -j4
```

# 参考资料：https://mnn-docs.readthedocs.io/en/latest/compile/tools.html#


# 2. 转换
# 在根目录下运行
./build/MNNConvert -f ONNX --modelFile ../onnx_block/embed.onnx --MNNModel ./build/mnn_block/embed.mnn --bizCode biz
./build/MNNConvert -f ONNX --modelFile ../onnx_block/lm.onnx --MNNModel ./build/mnn_block/lm.mnn --bizCode biz

for i in $(seq 0 27)
do
    echo $i
    inpname=../onnx_block/block_$i.onnx
    outname=./build/mnn_block/block_$i.mnn
    ./build/MNNConvert -f ONNX --modelFile $inpname --MNNModel $outname --bizCode biz
done


# 3. test
python ../tools/script/testMNNFromOnnx.py /home/xxx/proj/onnx_block/block_0.onnx


# 4. 量化
cd /path/to/MNN/build
cmake -DMNN_BUILD_QUANTOOLS=ON && make -j4
./quantized.out mobilnet.mnn mobilnet_quant.mnn mobilnet_quant.json
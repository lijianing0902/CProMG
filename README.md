# CProMG: Controllable Protein-Oriented Molecule Generation with Desired Binding Affinity and Drug-Like Properties

Deep learning-based molecule generation becomes a new paradigm of de novo molecule design since it enables fast and directional exploration in the vast chemical space. We elaborate a novel framework for controllable protein-oriented molecule generation. 
<!-- Here is the overview of the CProMG framework. -->

# Installation

| Main Package | Version |
| ------------ | ------- |
| Python       | 3.9.15  |
| Pytorch      | 3.9.15  |
| PyG          | 2.2.0   |
| CUDA         | 10.2    |
| RDKit        | 2022.03 |

# Datasets

<!-- 弄一个google drive放数据？ -->
# Training
```shell
python train.py --config  ./configs/CProMG-VQS.yml 
```
> 
# pretrained model
我们提供了两个模型，如下：
<!-- CProMG-VQS：./Pretrained/ CProMG-VQS.pt
CProMG-VQSLT: ./Pretrained/ CProMG-VQSLT.pt
运行如下代码为某一蛋白质口袋生成小分子（以CProMG-VQS模型为例）：
python gen.py --config ./configs/CProMG-VQS.yml --input ./data/test/1/pocket.pdb --model ./pretrained/CProMG-VQS.pt
--config 模型的配置文件，设置预期的分子属性
--input 蛋白质口袋路径 
--model 与训练模型路径 -->

# CProMG: Controllable Protein-Oriented Molecule Generation with Desired Binding Affinity and Drug-Like Properties

Deep learning-based molecule generation becomes a new paradigm of de novo molecule design since it enables fast and directional exploration in the vast chemical space. We elaborate a novel framework for controllable protein-oriented molecule generation. 
<!-- Here is the overview of the CProMG framework. -->

## Installation

We build the model in the following environment.

| Main Package | Version |
| ------------ | ------- |
| Python       | 3.9.15  |
| Pytorch      | 1.12.1  |
| PyG          | 2.2.0   |
| CUDA         | 10.2    |
| RDKit        | 2022.03 |

You can build your environment manually or through a yaml file.
### YAML file

```bash
conda env create -f env_CProMG.yml
conda activate CProMG
```
### Manually

```bash
# Create conda environment
conda create -n CProMG python=3.9
conda activate CProMG

# Install Pytorch (CUDA 10.2)
conda install pytorch==1.12.1 cudatoolkit=10.2 -c pytorch

# Install PyG (>=2.x)
conda install pyg -c pyg

# Install other toolkit


```

## Datasets

<!-- 弄一个google drive放数据？ -->

## Training

Use this command to train the model.
```bash
python train.py --config  ./configs/CProMG-VQS.yml 
```
You can also modify the configuration file to train the model with different parameters.

## Pretrained model

We provided the following two pre-trained models
- [CProMG-VQS](https://drive.google.com/file/d/1HlG2rnmcCYfP0gNjNpKAg4hhHiqEPWXn/view?usp=share_link)
- [CProMG-VQSLT](https://drive.google.com/file/d/1lWDAbYGpYuU2aO2Aa_rqIemmkwLbxNN4/view?usp=share_link)

Place the pretrained model (e.g. CProMG-VQS) under the `/Pretrained` folder and run the following code to generate small molecules for a protein pocket.

```bash
python gen.py --config ./configs/CProMG-VQS.yml --input ./data/test/1/pocket.pdb --model ./pretrained/CProMG-VQS.pt
```



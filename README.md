
This is the official implementation of our paper **Multimodal Fusion with LLM Content via Hierarchical Progressive Transformer for Explainable Fake News Detection**
##dataset
Original dataset from weibo21 and FineFake

Due to dataset ownership and privacy issues, we cannot make the full dataset publicly available, if you would like the dataset processed in a format that allows for easy reproduction of the code, please contact
2024112032016@stu.hznu.edu.cn

## Introduction
This repository provides the implementations of M<sup>3</sup>FEND and ten baseline models (BiGRU, TextCNN, RoBERTa, StyleLSTM, DualEmotion, EANN, EDDFN, MMoE, MoSE, MDFEND). Note that TextCNN and BiGRU are implemented with word2vec as word embedding in the original experiments, but we implement them with RoBERTa embedding in this repository.

## Requirements

- Python 3.6
- PyTorch > 1.0
- Pandas
- Numpy
- Tqdm


## Run

Parameter Configuration:

- dataset: the English or Chinese dataset, default for `ch`
- early_stop: default for `5`
- epoch: training epoches, default for `50`
- gpu: the index of gpu you will use, default for `0`
- lr: learning_rate, default for `0.0001`(en:0.0002)
- You can set the list of learning rates in grid_search.py's train_param
- 

You can run this code through:

```powershell
python main.py
```

## Reference



```

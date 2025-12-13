
This is the official implementation of **LLM-MFEFND**
##dataset
Original dataset from weibo21 and FineFake


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


This is the official implementation of **LLM-MFEFND**
##dataset
Original dataset from weibo21 and FineFake

Due to dataset ownership and privacy issues, we cannot make the full dataset publicly available, if you would like the dataset processed in a format that allows for easy reproduction of the code, please contact
2024112032016@stu.hznu.edu.cn

We plan to release the full code after the anonymous review process is completed, to avoid potential misuse and prevent any possible losses.

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

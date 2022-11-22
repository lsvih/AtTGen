# AtTGen

## Usage

```
usage: python3 main.py [-h] [--name NAME] [--do_train] [--do_eval]
               [--data_dir DATA_DIR] [--seed SEED] [--gpu_ids GPU_IDS]
               [--batch_size BATCH_SIZE] [--lr LR] [--epoch EPOCH]
               [--emb_dim EMB_DIM] [--encode_dim ENCODE_DIM]

configuration

optional arguments:
  -h, --help            show this help message and exit
  --name NAME           Experiment name, for logging and saving models
  --do_train            Whether to run traininog.
  --do_eval             Whether to run eval on the test set.
  --data_dir DATA_DIR   The input data dir.
  --seed SEED           The random seed for initialization
  --gpu_ids GPU_IDS     The GPU ids
  --batch_size BATCH_SIZE
                        Total batch size for training.
  --lr LR               The initial learning rate for Adam.
  --epoch EPOCH         Total number of training epochs to perform.
  --emb_dim EMB_DIM     The dimension of the embedding
  --encode_dim ENCODE_DIM
                        The dimension of the encoding
```

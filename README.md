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


## Dataset

Download the dataset to the `raw_data` folder, and run `python3 preprocess.py --dataset=xxxx` to preprocess the data.

> Using argument `--subject_guild True` to enable the subject guild function.

Pre-processed NYT dataset is attached in the `data` folder, which can be used directly.


## Usage

> What is available dataset by this model?
> Please preprocess the dataset, then check the files in `data` folder :)

### Train on CNShipNet dataset (Chinese datasets)

```bash
python3 main.py --gpu_ids=0 --data_dir=./data/CNShipNet/ --word_vocab=word_vocab.json --ontology_vocab=attribute_vocab.json --tokenizer=chn --do_train
```

### Train on NYT-style dataset (English datasets)

```bash
python3 main.py --gpu_ids=0 --data_dir=./data/nyt/ --ontology_vocab=relation_vocab.json --tokenizer=base --do_train
```

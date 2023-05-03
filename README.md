# AtTGen

Implementation of "AtTGen: Attribute Tree Generation for Real-World Attribute Joint Extraction", ACL 2023.

## Preparation

Please install the dependencies first:
```bash
pip install -r requirements.txt
```

## Usage

```
usage: main.py [-h] [--name NAME] [--do_train] [--do_eval]
               [--data_dir DATA_DIR] [--word_vocab WORD_VOCAB]
               [--ontology_vocab ONTOLOGY_VOCAB] [--tokenizer TOKENIZER]
               [--seed SEED] [--gpu_ids GPU_IDS] [--batch_size BATCH_SIZE]
               [--lr LR] [--epoch EPOCH] [--emb_dim EMB_DIM]
               [--encode_dim ENCODE_DIM] [--skip_subject SKIP_SUBJECT]

configuration

optional arguments:
  -h, --help            show this help message and exit
  --name NAME           Experiment name, for logging and saving models
  --do_train            Whether to run training.
  --do_eval             Whether to run eval on the test set.
  --data_dir DATA_DIR   The input data dir.
  --word_vocab WORD_VOCAB
                        The vocabulary file.
  --ontology_vocab ONTOLOGY_VOCAB
                        The ontology class file.
  --tokenizer TOKENIZER
                        The tokenizer type.
  --seed SEED           The random seed for initialization
  --gpu_ids GPU_IDS     The GPU ids
  --batch_size BATCH_SIZE
                        Total batch size for training.
  --lr LR               The initial learning rate for Adam.
  --epoch EPOCH         Total number of training epochs to perform.
  --emb_dim EMB_DIM     The dimension of the embedding
  --encode_dim ENCODE_DIM
                        The dimension of the encoding
  --skip_subject SKIP_SUBJECT
                        Whether to skip the subject
```

## Dataset

Download the dataset to the `raw_data` folder, and run `python3 preprocess.py --dataset=xxxx` to preprocess the data.

> Using argument `--subject_guild True` to enable the subject guild function.

Pre-processed NYT dataset is attached in the `data` folder, which can be used directly.

## Playground (Recommended)

Benefiting from the parameter efficiency of AtTGen, we can easily train and inference the model, and provide the trained model weights conveniently.

The trained model weights are in `runs/jave_best` file, which is trained by default hyper-parameters.

We use the sample data in [MEPAVE](https://github.com/jd-aig/JAVE/blob/master/data/jdai.jave.fashion.test.sample) to demonstrate the usage of AtTGen.

1. You can check the samples in `data/jave_sample` folder.
2. You can try this demonstration by directly running `python3 playground.py`.

## Usage

### Train on MEPAVE dataset

1. Preparing the MEPAVE dataset

Due to licensing restrictions, we cannot provide this dataset directly, please apply a license to use [here](https://github.com/jd-aig/JAVE),
 download the whole dataset and then put `*.txt` files in `raw_data/jave` folder.

2. Preprocessing the dataset

```bash
python3 preprocess.py --dataset=jave
```

3. Training the model

```bash
python3 main.py --do_train --gpu_ids=0 --data_dir=./data/jave/ --ontology_vocab=attribute_vocab.json --tokenizer=char --name=jave
```

4. Testing the model

```bash
python3 main.py --do_eval --gpu_ids=0 --data_dir=./data/jave/ --ontology_vocab=attribute_vocab.json --tokenizer=char --name=jave
```

### Train on CNShipNet dataset (Chinese datasets)

```bash
python3 main.py --gpu_ids=0 --data_dir=./data/CNShipNet/ --word_vocab=word_vocab.json --ontology_vocab=attribute_vocab.json --tokenizer=chn --do_train
```

### Train on NYT-style dataset (English datasets, e.g. AE100k)

```bash
python3 main.py --gpu_ids=0 --data_dir=./data/nyt/ --ontology_vocab=relation_vocab.json --tokenizer=base --do_train
```

## Reference

- [OpenJERE](https://github.com/WindChimeRan/OpenJERE) for Tree2Sequence Loader
- [JAVE](https://github.com/jd-aig/JAVE) for MEPAVE dataset
- [ScalingUp](https://github.com/lanmanok/ACL19_Scaling_Up_Open_Tagging/) for AE100k dataset

------

If you found this work useful, please cite it as follows:

```
@inproceedings{li-etal-2023-attgen,
    title = "AtTGen: Attribute Tree Generation for Real-World Attribute Joint Extraction",
    author = "Li, Yanzeng  and
      Xue, Bingcong  and
      Zhang, Ruoyu   and
      Zou, Lei",
    booktitle = "Proceedings of The 61st Annual Meeting of the Association for Computational Linguistics",
    month = july,
    year = "2023",
    address = "Toronto, Canada"
}
```
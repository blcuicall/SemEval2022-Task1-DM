# Cross-Attention Multitasking Framework for Definition Modeling

Source code for the paper **BLCU-ICALL at SemEval-2022 Task 1: Cross-Attention Multitasking Framework for Definition
Modeling** published on **NAACL 2022 Workshop**.

## Requirements

### Training & Evaluation Environment

- Pytorch
- tokenizers
- moverscore
- NLTK

In order to install them, you can run this command:

```
pip install -r requirements.txt
```

## Usage

1. Download CoDWoE Data from [CoDWoE Data Repository](https://codwoe.atilf.fr/)

2. Place the data file(s) anywhere you like, and modify `DATA_DIR` in the provided shell scripts for training and
   testing.

3. To train the model, simply use `train.sh` by:

```shell
./train.sh
```

4. To test the model, use the corresponding script `test-{lang}.sh`. English for example:

```shell
./test-en.sh
```

## Cite

```bibtex
@inproceedings{kong-etal-2022-semeval,
    title = "BLCU-ICALL at SemEval-2022 Task 1: Cross-Attention Multitasking Framework for Definition Modeling",
    author = "Kong, Cunliang and
        Wang, Yujie and
        Chong, Ruining and
        Yang, Liner and
        Zhang, Hengyuan and
        Yang, Erhong and
        Huang, Yaping",
    booktitle = "Proceedings of the 16th International Workshop on Semantic Evaluation (SemEval-2022)",
    year = "2022",
    publisher = "Association for Computational Linguistics",
}
```

## Contact

If you have questions, suggestions or bug reports, please email cunliang.kong@outlook.com

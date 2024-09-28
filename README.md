# Fine-tune UniSRec on BIM log data

In this project, we aim to fine-tune a sequential recommendation model [UniSRec](https://github.com/RUCAIBox/UniSRec), originally designed for e-commerce scenarios, on anonymized user tracking data from Allplan for next-command recommendations. 

**UniSRec** utilizes the associated description text of an item to learn *universal item representations* across different domains and platforms. Please refer to their paper for more information: https://arxiv.org/abs/2206.05941


## Setup env

```
# requirements
recbole>=1.1.1
python>=3.9.7
cudatoolkit>=11.3.1
pytorch>=1.11.0

# install conda env
conda env create -f environment.yml
conda activate unisrec
```
## Download the Pre-trained Model
We used the UniSRec model pretrained on five categories from Amazon review datasets (as reported in their paper).
The pretrained model can be downloaded from [[Google Drive](https://drive.google.com/drive/folders/1Uik0fMk4oquV_bS9lXTZuExAYbIDkEMW)]

Pretraining on Allplan command-related datasets may be considered in the future. But for now the document only considers presenting the workflow of fine tuning based on their model.

After unzipping, move `UniSRec-FHCKM-300.pth` to `saved/`.

## Getting started

### Preprocess the Allplan user tracking data
Fine-tuning on customized downstream datasets, you first need to preprocess your customized data.
Please refer to `dataset\README.md` for details.

### Train and evaluate on downstream datasets

After you preprocess your customized data, the files in the downstream datasets can be fine-tuned in two settings as follows

Fine-tune the pre-trained UniSRec model in transductive setting.

```
python finetune.py -d Allplan -p saved/UniSRec-FHCKM-300.pth
```

Fine-tune the pre-trained model in inductive setting.

```
python finetune.py -d Allplan -p saved/UniSRec-FHCKM-300.pth --train_stage=inductive_ft
```

Train UniSRec from scratch (w/o pre-training).

```
python finetune.py -d Allplan
```

Run baseline SASRec.

```
python run_baseline.py -m SASRec -d Allplan --config_files=props/finetune.yaml --hidden_size=300
```
Please refer to [[link]](https://github.com/RUCAIBox/UniSRec/issues/4#issuecomment-1316045022) for more scripts of utlized for baselines.

### Results

We provide 2 fine-tuned models weights `UniSRec-Sep-06-2024_16-34-48.pth` and `UniSRec-Sep-10-2024_21-32-33` for allplan data, which can be downloaded from [Syncandshare](https://syncandshare.lrz.de/getlink/fiKVL8z1GxTcvkQadJpmrX/).
The results of both expertiments can be found in `log/UniSRec` and `log_tensorboard`.

## Acknowledgement

The implementation is based on the [UniSRec](https://github.com/RUCAIBox/UniSRec)

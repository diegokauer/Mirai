# Mirai: Mammography-based model for breast cancer risk [![DOI](https://zenodo.org/badge/315745008.svg)](https://zenodo.org/badge/latestdoi/315745008)

# Introduction
This repository was used to develop Mirai, the risk model described in: [Towards Robust Mammography-Based Models for Breast Cancer Risk](https://www.science.org/doi/10.1126/scitranslmed.aba4373). Mirai was designed to predict risk at multiple time points, leverage potentially missing risk-factor information, and produce predictions that are consistent across mammography machines. Mirai was trained on a large dataset from Massachusetts General Hospital (MGH) in the US and was tested on held-out test sets from MGH, Karolinska in Sweden and Chang Gung Memorial Hospital in Taiwan, obtaining C-indices of 0.76 (0.74, 0.80), 0.81 (0.79, 0.82), 0.79 (0.79, 0.83), respectively. Mirai obtained significantly higher five-year ROC AUCs than the Tyrer-Cuzick model (p<0.001) and prior deep learning models, Hybrid DL (p<0.001) and ImageOnly DL (p<0.001), trained on the same MGH dataset. In our paper, we also demonstrate that Mirai was more significantly accurate in identifying high risk patients than prior methods across all datasets. On the MGH test set, 41.5% (34.4, 48.5) of patients who would develop cancer within five-years were identified as high risk, compared to 36.1% (29.1, 42.9) by Hybrid DL (p=0.02) and 22.9% (15.9, 29.6) by Tyrer-Cuzick lifetime risk (p<0.001).

This code base is meant to achieve a few goals:
- Provide exact implementation details for the development of Mirai
- Help researchers to install Mirai for clinical workflows
- Help researchers to validate Mirai on large datasets
- Help researchers to fine-tune Mirai on large datasets

We note that this code-base is an extension of [OncoNet](https://github.com/yala/OncoNet_Public), which we used to develop Hybrid DL and ImageOnly DL.
Mirai is designed to operate in bulk, on PNG images. For a conversion toolkit, see [OncoData](https://github.com/reginabarzilaygroup/OncoData_Public)

Also see the [wiki](https://github.com/reginabarzilaygroup/Mirai/wiki) for more information.

## Aside on Software Dependencies
This code assumes python3.8 and a Linux environment.
The package can be installed with pip:

```bash
git clone https://github.com/reginabarzilaygroup/Mirai
cd Mirai
pip install -e .
```

If you are familiar with docker, you can also directly leverage the [Ark docker container](https://hub.docker.com/r/mitjclinic/mirai) which has all the dependencies preinstalled and the trained Mirai model (see below).

## Preprocessing
Our code-base operates on PNG images. We converted presentation view dicoms to PNG16 files using the DCMTK library. We used the dcmj2pnm program (v3.6.1, 2015) with +on2 and–min-max-window flags. To this, you can use DCMTK directly or [OncoData](https://github.com/yala/OncoData_Public), our python wrapper for converting dicoms.

## Inclusion Criteria
Mirai expects all four standard “For Presentation” views of the mammogram.  Specifically, it requires an “L CC”, “L MLO”, “R CC”, “R MLO”. It will not work without all four views or with “For Processing” mammograms.  As a result, we unfortunately cannot provide risk assessments for patients with only non-standard views or unilateral mammograms. Moreover, we cannot run the model on marked up images (i.e. images with some CAD or human annotations).  All of the mammograms used in our study were captured using either the Hologic Selenia or Selenia Dimensions mammography devices. We have yet tested Mirai on other machines.

# Reproducing Mirai
As described in the supplementary material of the paper, Mirai was trained in two phases; first, we trained the image encoder in conjunction with the risk factor predictor and additive hazard layer to predict breast cancer independently from each view without using conditional adversarial training. In this stage, we initialized our image encoder with weights from ImageNet, and augmented our training set with random flips and rotations of the original images. We found that adding an adversarial loss at this stage or training the whole architecture end-to-end prevented the model from converging. In the second stage of training, we froze our image encoder, and trained the image aggregation module, the risk factor prediction module, the additive hazard layer, and the device discriminator in a conditional adversarial training regime. We trained our adversary for three steps for every one step of training Mirai. In each stage, we performed small hyperparameter searches and chose the model that obtained the highest C-index on the development set.

The grid searches are shown in :

`configs/mirai_base.json` and `configs/mirai_full.json`

The grid searches were run using our job-dispatcher, as shown bellow.

`python scripts/dispatcher.py --experiment_config_path configs/mirai_base.json --result_path mirai_base_sweep.csv`

We selected the image encoder with the highest C-index on the development set, and leveraged it for the second stage hyperparameter sweep.

`python scripts/dispatcher.py --experiment_config_path configs/mirai_full.json --result_path mirai_full_sweep.csv`

We note that this command run relies on integrations that were specific to the MGH data, and so the exact line above will not run on your system. The configs above are meant to specify exact implementation details and our experimental procedure.

# Using Mirai
Mirai (the trained model) and all code are released under the MIT license. 

## Installing Mirai for clinical use

It is recommended to use [Ark](https://github.com/reginabarzilaygroup/ark), our framework for deploying mammography-based models in the clinic. Ark can be easily installed on premise using Docker, and it provides a simple HTTP interface to get risk assessments for a given patient's dicom files. Ark encapsulates all the dependencies and necessary preprocessing.

Alternatively, one can use the following command to run Mirai directly:

```bash
python3 onconet/predict.py ccl1.dcm ccr1.dcm mlol2.dcm mlor2.dcm
```

Where `ccl1.dcm`, `ccr1.dcm`, `mlol2.dcm`, `mlor2.dcm` are the paths to the dicom files for the left CC, right CC, left MLO, and right MLO views of the mammogram, respectively. The output will be a risk assessment for the patient in the form of a JSON file.

See `run_predict_demo.sh` for a longer example

### How validate the model on a large dataset
To validate Mirai, you can use the following command: `sh demo/validate.sh`
The full bash command (inside the validate.sh file) is:
```
python scripts/main.py  --model_name mirai_full --img_encoder_snapshot snapshots/mgh_mammo_MIRAI_Base_May20_2019.p --transformer_snapshot snapshots/mgh_mammo_cancer_MIRAI_Transformer_Jan13_2020.p  --callibrator_snapshot snapshots/callibrators/MIRAI_FULL_PRED_RF.callibrator.p --batch_size 1 --dataset csv_mammo_risk_all_full_future --img_mean 7047.99 --img_size 1664 2048 --img_std 12005.5 --metadata_path demo/sample_metadata.csv --test --prediction_save_path demo/validation_output.csv
```

Alternatively, you could launch the same validation script using our job-dispatcher with the following command:
```
python scripts/dispatcher.py --experiment_config_path configs/validate_mirai.json --result_path finetune_results.csv
```

What you need to validate the model:
- Install the dependencies (see above)
- Get access to the snapshot files (in snapshots folder of docker container)
- Convert your dicoms to PNGs (see above)
- Create a CSV file describing your dataset. For an example, see `demo/sample_metadata.csv`. We note that all the columns are required.
    - `patient_id`: ID string for this patient. Is used to link together mammograms for one patient.
    - `exam_id`: ID string for this mammogram. Is used to link together several files for one mammogram. Note, this code-base assumes that "patient_id + exam_id" is the unique key for a mammogram.
    - `laterality`: Laterality of the mammogram. Can only take values 'R' or 'L' for right and left.
    - `view`: View of the dicom. Can only take values 'MLO' or 'CC'. Other views are not supported.
    - `file_path`: Absolute path to the PNG16 image for this view of the mammogram
    - `years_to_cancer`: Integer the number of years from this mammogram that the patient was diagnosed with breast cancer. If the patient doesn't develop cancer during the observed data, enter 100. If the cancer was found on this mammogram, enter 0.
    - `years_to_last_followup`: Integer reflecting how many years from the mammogram we know the patient is cancer free. For example, if a patient had a negative mammogram in 2010 (and this row corresponds to that mammogram), and we have negative followup until 2020, then enter 10.
    - `split_group`: Can take values `train`, `dev` or `test` to note the training, validation and testing samples.

Before running `validate.sh`, make sure to replace `demo/sample_metadata.csv` with the path to your metadata path and to replace `demo/validation_output.csv` to wherever you want predictions will be saved.

After running `validate.sh`, our code-base will print out the AUC for each time-point and save the predictions for each mammogram in `prediction_save_path`. For an example of the output file format, see `demo/validation_output.csv`. The key `patient_exam_id` is defined as `patient_id \tab exam_id`.

### How to fine-tune the model
To finetune Mirai, you can use the following commands: `sh demo/finetune.sh`
The full bash command (inside the validate.sh file) is:

```
python scripts/dispatcher.py --experiment_config_path configs/fine_tune_mirai.json --result_path finetune_results.csv
```

It creates a grid search over possible fine-tuning hyperparameters (see `configs/finetune_mirai.json`) and launches jobs across the available GPUs (as defined in `available_gpus`). The results will be summarized in `finetune_results.csv` or wherever you set `results_path`. We note that each job launches just a shell command. By editing `configs/finetune_mirai.json` or creating your own config json file, you can explore any hyperparameters or architecture supported in the code base.

What finetune the model, you will need the same dependencies, preprocessing and CSV file as listed above to validate Mirai. We recommend you first evaluate Mirai before you try to finetune it.


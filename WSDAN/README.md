# DigitalMedicine_Case2
This repository is about Case2 in the class Digital Medicine, NYCU. The main target is to recognize the x-ray images of pneumonia.
# Environment
```
python==3.8.5
pytorch==1.9.1
```
# Prepare Dataset
You can download the dataset from codalab in-class competition:
```
https://www.kaggle.com/c/digital-medicine-2021-case-presentation-2/data
```
After downloading, run this command to collect images:
```
python3 move.py
```
# Training
To train the model, run this command:
```
python3 train.py --net model_name
```
You can choose inception_v3, resnet50, resnet101, resnet152 as the backbone model.
# Evaluation
If you want to evaluate a single model, run this command:
```
python3 single_model_eval.py --net model_name
```
# Result
```
Mean F-score: 0.64761
```

# Digital-Medicine-Case-Presentation2
數位醫學Case2的code，詳細情形可參考[Case2](https://www.kaggle.com/c/digital-medicine-2021-case-presentation-2)

## 環境
- CUDA 10.1
- python 3.6

## 重現方法
跟著以下步驟可以重現同樣成果：
1. [下載作業檔案](#下載作業檔案)
2. [準備資料](#準備資料)
3. [準備model](#準備model)
4. [訓練model](#訓練model)
5. [重現結果](#重現結果)

## 下載作業檔案
```
git clone https://github.com/nomiaro/Digital-Medicine.git
```

## 準備資料
前往[Kaggle](https://www.kaggle.com/c/digital-medicine-2021-case-presentation-2/data)將data.zip下載下來並在Case2資料夾解壓縮
```
cd Digital-Medicine/Case2
unzip data.zip
```
完成後擺放成以下結構:
```
Case2
  +- DCM2JPG.py
  +- ...
  +- data
  |  +- train
  |  |  +- ...dcm
  |  +- ...
```
用`preparedata.py`將資料移出多餘的資料夾
```
python preparedata.py
```
用`DCM2JPG.py`將DCM轉成JPD存入data/{}_images資料夾
```
pip install pydicom
python DCM2JPG.py
```
用`prepare_anno.py`準備annotation檔案
```
pip install pandas
python prepare_anno.py
```

## 準備model
下載mmclassification
```
# 下載對應CUDA版本的PyTorch
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
git clone https://github.com/open-mmlab/mmclassification.git
pip install mmcv-full
pip install -e .
pip install mmcls
```
將`filelist.py`上傳到`mmclassification/mmcls/datasets/`裡
```
mv filelist.py mmclassification/mmcls/datasets/
```
修改`mmclassification/mmcls/datasets/__init__.py`中datasets引用的相關設定:
```python
from .base_dataset import BaseDataset
...
from .filelist import Filelist

__all__ = [
    'BaseDataset', ... ,'Filelist'
]
```
將`swin_transformer_finetune.py`移動到`mmclassification/configs/`
```
mv swin_transformer_finetune.py mmclassification/configs/
```

## 訓練model
```
cd mmclassification
python tools/train.py configs/swin_transformer_finetune.py --work-dir {your_result_dir} --seed 2021
```

## 重現結果
下載[checkpoint](https://drive.google.com/file/d/17d96qH1GtNhpBzJeyhZWCNJlRBdh5eTl/view?usp=sharing)
在`mmclassification/tool/test.py:164`加上:
```python
CLASSES = ['Negative', 'Typical', 'Atypical']
```
產生預測結果
```
cd mmclassification
python tools/test.py configs/swin_transformer_finetune.py {your_checkpoint} --metrics f1_score --out {filename.pkl}
```
將`pkl2csv.py`移到`mmclassification/`，並用`pkl2csv.py`產生同pkl名稱的csv檔
```
mv ../pkl2csv.py ./
python pkl2csv.py {filename.pkl}
```

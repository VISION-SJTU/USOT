# Preprocessing VID (Object detection from video)

The preprocessing procedure for Large Scale Visual Recognition Challenge 2015 (ILSVRC2015).

## Tutorial

Download the raw VID dataset and do some pre-works.

```shell
wget http://bvisionweb1.cs.unc.edu/ilsvrc2015/ILSVRC2015_VID.tar.gz
tar -xzvf ./ILSVRC2015_VID.tar.gz
ln -sfb $PWD/ILSVRC2015/Annotations/VID/train/ILSVRC2015_VID_train_0000 ILSVRC2015/Annotations/VID/train/a
ln -sfb $PWD/ILSVRC2015/Annotations/VID/train/ILSVRC2015_VID_train_0001 ILSVRC2015/Annotations/VID/train/b
ln -sfb $PWD/ILSVRC2015/Annotations/VID/train/ILSVRC2015_VID_train_0002 ILSVRC2015/Annotations/VID/train/c
ln -sfb $PWD/ILSVRC2015/Annotations/VID/train/ILSVRC2015_VID_train_0003 ILSVRC2015/Annotations/VID/train/d
ln -sfb $PWD/ILSVRC2015/Annotations/VID/val ILSVRC2015/Annotations/VID/train/e

ln -sfb $PWD/ILSVRC2015/Data/VID/train/ILSVRC2015_VID_train_0000 ILSVRC2015/Data/VID/train/a
ln -sfb $PWD/ILSVRC2015/Data/VID/train/ILSVRC2015_VID_train_0001 ILSVRC2015/Data/VID/train/b
ln -sfb $PWD/ILSVRC2015/Data/VID/train/ILSVRC2015_VID_train_0002 ILSVRC2015/Data/VID/train/c
ln -sfb $PWD/ILSVRC2015/Data/VID/train/ILSVRC2015_VID_train_0003 ILSVRC2015/Data/VID/train/d
ln -sfb $PWD/ILSVRC2015/Data/VID/val ILSVRC2015/Data/VID/train/e
```

Prepare the flow estimation module (ARFlow).
Download the pretrained model of ARFlow `pwclite_ar_mv.tar` 
from [Google Drive](https://drive.google.com/drive/folders/1oa5fJN_QicIF1aJ-Uth2IQaY_bOW49Ia?usp=sharing)
in folder `pretrain/preprocessing` and place it in 
`$USOT_PATH/preprocessing/flow_module/checkpoint`.

Install the Flow Module. We use pytorch 1.1.0 + CUDA-10.0 to build this module.
 
```shell
cd $USOT_PATH
bash ./preprocessing/install_preprocessing.sh $CONDA_PATH USOTPre
source activate USOTPre && export PYTHONPATH=$(pwd)
cd $USOT_PATH/preprocessing/flow_module/models/correlation_package
python setup.py install
```

Generate pseudo boxes for VID with DP + Flow. 
 Refer to source code for detailed options. 
 
```shell
cd $USOT_PATH/preprocessing/datasets_train/vid
python parse_vid_flow.py
```

Crop dataset like SiamFC. 
 Refer to source code for detailed options.
 
```shell
python par_crop.py
```

Generate pseudo annotation file for training.
 Refer to source code for detailed options.
 
```shell
python gen_json.py
```

Note: We have provided two shortcuts for preprocessing. 
Refer to `$USOT_PATH/README.md` for details.
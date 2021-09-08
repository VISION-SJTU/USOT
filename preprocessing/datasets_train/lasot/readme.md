# Preprocessing LaSOT (train)

The preprocessing procedures for LaSOT.

## Tutorial

Download the raw LaSOT dataset. 
DO NOT FORGET to place the `testing_set.txt` in the lasot dataset folder.

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

Generate pseudo boxes for LaSOT with DP + Flow. 
 Refer to source code for detailed options.
 
```shell
cd $USOT_PATH/preprocessing/datasets_train/lasot
python parse_lasot_flow.py
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

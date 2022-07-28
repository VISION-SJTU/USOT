# Unsupervised Single Object Tracking (USOT)

:herb: **[Learning to Track Objects from Unlabeled Videos](https://arxiv.org/abs/2108.12711)**

Jilai Zheng, Chao Ma, Houwen Peng and Xiaokang Yang

2021 IEEE/CVF International Conference on Computer Vision (ICCV)

## Introduction

 This repository implements unsupervised deep tracker USOT, 
 which learns to track objects from unlabeled videos. 
 
 Main ideas of USOT are listed as follows.

 - Coarsely discovering **moving objects** from videos, with pseudo boxes precise enough for **bbox regression**. 
 - Training a **naive Siamese tracker** from single-frame pairs, then gradually extending it to **longer temporal spans**.
 - Following **cycle memory** training paradigm, enabling unsupervised tracker to **update online**.

## Results

 Results of USOT and USOT* on recent tracking benchmarks.
 
 <table>
  <tr>
    <th>Model</th>
    <th>VOT2016<br> EAO </th>
    <th>VOT2018<br> EAO </th>
    <th>VOT2020<br> EAO </th>
    <th>LaSOT<br> AUC (%)</th>
    <th>TrackingNet<br> AUC (%)</th>
    <th>OTB100<br> AUC (%)</th>
    <th>GOT10k<br> AO (%)</th>
  </tr>
  <tr>
    <td>USOT</td>
    <td>0.351</td>
    <td>0.290</td>
    <td>0.222</td>
    <td>33.7</td>
    <td>59.9</td>
    <td>58.9</td>
    <td>0.444</td>
  </tr>
  <tr>
    <td>USOT*</td>
    <td>0.402</td>
    <td>0.344</td>
    <td>0.219</td>
    <td>35.8</td>    
    <td>61.5</td>
    <td>57.4</td>
    <td>0.441</td>
  </tr>
 </table>
 
 Raw result files can be found in folder `result` 
 from [Google Drive](https://drive.google.com/drive/folders/1oa5fJN_QicIF1aJ-Uth2IQaY_bOW49Ia?usp=sharing).

## Tutorial

### Environments

 The environment we utilize is listed as follows.

 - **Preprocessing**: Pytorch 1.1.0 + CUDA-9.0 / 10.0 (following ARFlow)
 - **Train / Test / Eval**: Pytorch 1.7.1 + CUDA-10.0 / 10.2 / 11.1
 
 If you have problems for preprocessing, you can actually skip it by downloading off-the-shelf 
 preprocessed materials.

### Preparations

 Assume the project root path is `$USOT_PATH`. 
 You can build an environment for development with the provided script, 
 where `$CONDA_PATH` denotes your anaconda path. 

 ```shell
 cd $USOT_PATH
 bash ./preprocessing/install_model.sh $CONDA_PATH USOT
 source activate USOT && export PYTHONPATH=$(pwd)
 ```

 You can revise the CUDA toolkit version for pytorch in `install_model.sh` (by default 10.0). 

### Test and Eval

 First, we provide both models utilized in our paper (`USOT.pth` and `USOT_star.pth`).
 You can download them in folder `snapshot` from [Google Drive](https://drive.google.com/drive/folders/1oa5fJN_QicIF1aJ-Uth2IQaY_bOW49Ia?usp=sharing), 
 and place them in `$USOT_PATH/var/snapshot`.

 Next, you can link your wanted benchmark dataset (e.g. VOT2018) to `$USOT_PATH/datasets_test` as follows.
 The ground truth json files for some benchmarks (e.g `VOT2018.json`) 
 can be downloaded in folder `test` from [Google Drive](https://drive.google.com/drive/folders/1oa5fJN_QicIF1aJ-Uth2IQaY_bOW49Ia?usp=sharing), 
 and placed also in `$USOT_PATH/datasets_test`.

 ```shell
 cd $USOT_PATH && mkdir datasets_test
 ln -s $your_benchmark_path ./datasets_test/VOT2018
 ```

 After that, you can test the tracker on these benchmarks (e.g. VOT2018) as follows. 
 The raw results will be placed in `$USOT_PATH/var/result/VOT2018/USOT`.

 ```shell
 cd $USOT_PATH
 python -u ./scripts/test_usot.py --dataset VOT2018 --resume ./var/snapshot/USOT_star.pth
 ```

 The inference result can be evaluated with pysot-toolkit. Install pysot-toolkit before evaluation.
 ```shell
 cd $USOT_PATH/lib/eval_toolkit/pysot/utils
 python setup.py build_ext --inplace
 ```
 
 Then the evaluation can be conducted as follows.
 ```shell
 cd $USOT_PATH
 python ./lib/eval_toolkit/bin/eval.py --dataset_dir datasets_test \
         --dataset VOT2018 --tracker_result_dir var/result/VOT2018 --trackers USOT
 ```

### Train

 First, download the pretrained backbone in folder `pretrain` 
 from [Google Drive](https://drive.google.com/drive/folders/1oa5fJN_QicIF1aJ-Uth2IQaY_bOW49Ia?usp=sharing) 
 into `$USOT_PATH/pretrain`. 
 Note that USOT* and USOT are respectively trained from `imagenet_pretrain.model` and `moco_v2_800.model`.

 Second, preprocess the raw datasets with the paradigm of DP + Flow. 
 Refer to `$USOT_PATH/preprocessing/datasets_train` for details. 

 In fact, we have provided two shortcuts for skipping this preprocessing procedure. 
 - You can directly download the generated pseudo box files (e.g. `got10k_flow.json`) 
 in folder `train/box_sample_result` 
 from [Google Drive](https://drive.google.com/drive/folders/1oa5fJN_QicIF1aJ-Uth2IQaY_bOW49Ia?usp=sharing), 
 and place them into the corresponding dataset preprocessing path 
 (e.g. `$USOT_PATH/preprocessing/datasets_train/got10k`), in order to skip the box generation procedure.
 - You can directly download the whole cropped training dataset (e.g. `got10k_flow.tar`) in 
 dataset folder from [Baidu Drive (6887)](https://pan.baidu.com/s/1xsOuLKanK-w21xKePk4PLw) (e.g. `train/GOT-10k`), 
 which enables you to skip all procedures in preprocessing.
 
 Third, revise the config file for training as `$USOT_PATH/experiments/train/USOT.yaml`.
 Very important options are listed as follows.
 - GPUS: the gpus for training, e.g. '0,1,2,3'
 - TRAIN/PRETRAIN: the pretrained backbone, e.g. 'imagenet_pretrain.model'
 - DATASET: the folder for your cropped training instances and their pseudo annotation files, 
      e.g. PATH: '/data/got10k_flow/crop511/', ANNOTATION: '/data/got10k_flow/train.json'

 Finally, you can start the training phase with the following script.
 The training checkpoints will also be placed automatically in `$USOT_PATH/var/snapshot`.

 ```shell
 cd $USOT_PATH
 python -u ./scripts/train_usot.py --cfg experiments/train/USOT.yaml --gpus 0,1,2,3 --workers 32
 ```

 We also provide a onekey script for train, test and eval. 

 ```shell
 cd $USOT_PATH
 python ./scripts/onekey_usot.py --cfg experiments/train/USOT.yaml
 ```

## Citation

 If any parts of our paper and codes are helpful to your work, please generously citing:
 
 ```
@inproceedings{zheng-iccv2021-usot,
    title={Learning to Track Objects from Unlabeled Videos},
    author={Jilai Zheng and Chao Ma and Houwen Peng and Xiaokang Yang},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
    year={2021}
}
```

## Reference
 
 We refer to the following repositories when implementing our unsupervised tracker. Thanks for their great work.
 
 - [lliuz/ARFlow](https://github.com/lliuz/ARFlow)
 - [researchmm/TracKit](https://github.com/researchmm/TracKit)
 - [vacancy/PreciseRoIPooling](https://github.com/vacancy/PreciseRoIPooling)
 - [facebookresearch/moco](https://github.com/facebookresearch/moco)

## Contact
 
  Feel free to contact me if you have any questions.
 
 - Jilai Zheng, email: [zhengjilai@sjtu.edu.cn](https://github.com/zhengjilai)


#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "ERROR! Illegal number of parameters. Usage: bash install_model.sh conda_install_path environment_name"
    exit 0
fi

conda_install_path=$1
conda_env_name=$2

source $conda_install_path/etc/profile.d/conda.sh
echo "****************** Creating conda environment ${conda_env_name} python=3.7 ******************"
conda create -y -n $conda_env_name python=3.7

echo ""
echo ""
echo "****************** Activating conda environment ${conda_env_name} ******************"
conda activate $conda_env_name

echo ""
echo ""
echo "****************** Installing pytorch 1.7.1 with cuda 10.0 ******************"
echo "****************** Revise here to install pytorch with cuda 11.0/11.1 for RTX3090 ******************"
conda install -y pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.0 -c pytorch

echo ""
echo ""
echo "****************** Installing matplotlib 2.2.2 ******************"
conda install -y matplotlib=2.2.2

echo ""
echo ""
echo "****************** Installing pandas ******************"
conda install -y pandas

echo ""
echo ""
echo "****************** Installing opencv ******************"
pip install opencv-python

echo ""
echo ""
echo "****************** Installing tensorboardX ******************"
pip install tensorboardX

echo ""
echo ""
echo "****************** Installing cython ******************"
conda install -y cython

echo ""
echo ""
echo "****************** Installing skimage ******************"
pip install scikit-image

echo ""
echo ""
echo "****************** Installing pillow ******************"
pip install 'pillow<7.0.0'

echo ""
echo ""
echo "****************** Installing scipy ******************"
pip install scipy

echo ""
echo ""
echo "****************** Installing shapely ******************"
pip install shapely

echo ""
echo ""
echo "****************** Installing easydict ******************"
pip install easydict

echo ""
echo ""
echo "****************** Installing imgaug ******************"
pip install imgaug

echo ""
echo ""
echo "****************** Installing other useful packages ******************"
pip install jpeg4py 
pip install mpi4py
pip install pyyaml
pip install tqdm
pip install colorama
pip install numba

echo "****************** Installation complete! ******************"

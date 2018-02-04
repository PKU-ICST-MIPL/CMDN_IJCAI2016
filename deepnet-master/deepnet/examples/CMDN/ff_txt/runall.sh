#!/bin/bash
# Trains a feed forward net on MNIST.
train_deepnet=${deepnet}/trainer.py
ff_txt_dir=${deepnet}/examples/CMDN/ff_txt
python ${train_deepnet} ${ff_txt_dir}/model.pbtxt ${ff_txt_dir}/train.pbtxt ${ff_txt_dir}/eval.pbtxt

sh ff_txt/extract_reps.sh

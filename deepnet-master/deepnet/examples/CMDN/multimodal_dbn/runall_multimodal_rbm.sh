#!/bin/bash

gpu_mem=4G
main_mem=30G

mdbn_dir=${deepnet}/examples/CMDN/multimodal_dbn
prefix=${mdbn_dir}/data

trainer=${deepnet}/trainer.py
extract_rep=${deepnet}/extract_rbm_representation.py
model_output_dir=${mdbn_dir}/data/dbn_models
data_output_dir=${mdbn_dir}/data/dbn_reps

models_dir=${mdbn_dir}/models
trainers_dir=${mdbn_dir}/trainers/dbn

clobber=false

mkdir -p ${model_output_dir}
mkdir -p ${data_output_dir}

# MERGE IMAGE AND TEXT DATA PBTXT FOR TRAINING JOINT RBM
#if ${clobber} || [ ! -e ${data_output_dir}/joint_rbm_LAST/input_data.pbtxt ]; then
  mkdir -p ${data_output_dir}/joint_rbm_LAST
  python ${mdbn_dir}/merge_dataset_pb.py \
      ${data_output_dir}/image_rbm2_LAST/data.pbtxt \
      ${data_output_dir}/text_rbm2_LAST/data.pbtxt \
      ${data_output_dir}/joint_rbm_LAST/input_data.pbtxt || exit 1
#fi

#if ${clobber} || [ ! -e ${model_output_dir}/joint_layer2_LAST ]; then
  echo "Training joint layer."
  python ${trainer} ${models_dir}/joint_rbm.pbtxt \
    ${trainers_dir}/train_CD_joint_layer.pbtxt ${mdbn_dir}/eval.pbtxt || exit 1
#fi

#if ${clobber} || [ ! -e ${data_output_dir}/joint_layer2_generated_text/data.pbtxt ]; then
  echo "Inferring missing text"
  python ${mdbn_dir}/scripts/sample_text.py ${model_output_dir}/joint_rbm_LAST \
    ${trainers_dir}/train_CD_joint_layer.pbtxt ${data_output_dir}/joint_layer2_generated_text \
    ${data_output_dir}/joint_rbm_LAST/input_data.pbtxt ${gpu_mem} ${main_mem} || exit 1
#fi

#if ${clobber} || [ ! -e ${data_output_dir}/joint_layer2_generated_image/data.pbtxt ]; then
  echo "Inferring missing image"
  python ${mdbn_dir}/scripts/sample_image.py ${model_output_dir}/joint_rbm_LAST \
    ${trainers_dir}/train_CD_joint_layer.pbtxt ${data_output_dir}/joint_layer2_generated_image \
    ${data_output_dir}/joint_rbm_LAST/input_data.pbtxt ${gpu_mem} ${main_mem} || exit 1
#fi

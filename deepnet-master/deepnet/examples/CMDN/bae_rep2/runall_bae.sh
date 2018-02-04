#!/bin/bash

gpu_mem=4G
main_mem=25G

bae_dir=${deepnet}/examples/CMDN/bae_rep2

trainer=${deepnet}/trainer.py
extract_rep=${deepnet}/extract_rbm_representation.py
extract_rep_nn=${deepnet}/extract_neural_net_representation.py
model_output_dir=${bae_dir}/data/ae_models
data_output_dir=${bae_dir}/data/ae_reps

models_dir=${bae_dir}/ae_models
trainers_dir=${bae_dir}/ae_trainers

clobber=false

mkdir -p ${model_output_dir}
mkdir -p ${data_output_dir}

echo BAEs
#if ${clobber} || [ ! -e ${model_output_dir}/bae_LAST ]; then
  #python gen_bae_data.py || exit 1
python ${trainer} ${bae_dir}/bae.pbtxt \
  ${bae_dir}/train_ae_layer.pbtxt ${bae_dir}/eval.pbtxt || exit 1
python ${extract_rep_nn} ${model_output_dir}/bae_LAST \
  ${bae_dir}/train_ae_layer.pbtxt ${data_output_dir}/bae_LAST \
  None image_tied_hidden text_tied_hidden || exit 1
#fi


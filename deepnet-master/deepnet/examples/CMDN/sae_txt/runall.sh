#!/bin/bash
# Trains DBN on MNIST.

gpu_mem=4G
main_mem=25G

sae_txt_dir=${deepnet}/examples/CMDN/sae_txt

train_deepnet=${deepnet}/trainer.py
extract_rep_nn=${deepnet}/extract_neural_net_representation.py
model_output_dir=${sae_txt_dir}/data/sae_models
data_output_dir=${sae_txt_dir}/data/sae_reps

model_dir=${sae_txt_dir}/models

mkdir -p ${model_output_dir}
mkdir -p ${data_output_dir}

echo "Autoencoder 1"
python ${train_deepnet} ${model_dir}/model_layer1.pbtxt ${sae_txt_dir}/train.pbtxt ${sae_txt_dir}/eval.pbtxt || exit 1
python ${extract_rep_nn} ${model_output_dir}/mnist_autoencoder_relu_LAST ${sae_txt_dir}/train.pbtxt ${data_output_dir}/mnist_autoencoder_relu_LAST None hidden1 || exit 1
echo "Autoencoder 2"
python ${train_deepnet} ${model_dir}/model_layer2.pbtxt ${sae_txt_dir}/train.pbtxt ${sae_txt_dir}/eval.pbtxt || exit 1
python ${extract_rep_nn} ${model_output_dir}/mnist_autoencoder_relu_2layer_LAST ${sae_txt_dir}/train.pbtxt ${data_output_dir}/mnist_autoencoder_relu_2layer_LAST None hidden2 || exit 1
#echo "Classifier"
#${train_deepnet} classifier.pbtxt train.pbtxt eval.pbtxt


python mat2npy.py

echo "run SAE for image"
sh sae_img/runall.sh
echo "run SAE for text"
sh sae_txt/runall.sh

echo "run RBM for image"
sh multimodal_dbn/runall_dbn_img.sh
echo "run RBM for text"
sh multimodal_dbn/runall_dbn_txt.sh
echo "run multimodal DBN"
sh multimodal_dbn/runall_multimodal_rbm.sh

echo "process result of SAE and multimodal DBN for joint"
python norm_sae.py
cp multimodal_dbn/data/dbn_reps/joint_layer2_generated_text/train/joint_hidden-00001-of-00001.npy joint_img/data/joint_img/train/
cp multimodal_dbn/data/dbn_reps/joint_layer2_generated_text/test/joint_hidden-00001-of-00001.npy joint_img/data/joint_img/test/
cp multimodal_dbn/data/dbn_reps/joint_layer2_generated_text/validation/joint_hidden-00001-of-00001.npy joint_img/data/joint_img/validation/
cp multimodal_dbn/data/dbn_reps/joint_layer2_generated_image/train/joint_hidden-00001-of-00001.npy joint_txt/data/joint_txt/train/
cp multimodal_dbn/data/dbn_reps/joint_layer2_generated_image/test/joint_hidden-00001-of-00001.npy joint_txt/data/joint_txt/test/
cp multimodal_dbn/data/dbn_reps/joint_layer2_generated_image/validation/joint_hidden-00001-of-00001.npy joint_txt/data/joint_txt/validation/

echo "run joint_rbm for image"
sh joint_img/runall.sh
echo "run joint_rbm for text"
sh joint_txt/runall.sh

cp joint_img/joint_reps/joint_img_LAST/train/joint_img_hidden-00001-of-00001.npy ff_img/data/train/
cp joint_img/joint_reps/joint_img_LAST/test/joint_img_hidden-00001-of-00001.npy ff_img/data/test/
cp joint_img/joint_reps/joint_img_LAST/validation/joint_img_hidden-00001-of-00001.npy ff_img/data/validation/
cp joint_txt/joint_reps/joint_txt_LAST/train/joint_txt_hidden-00001-of-00001.npy ff_txt/data/train/
cp joint_txt/joint_reps/joint_txt_LAST/test/joint_txt_hidden-00001-of-00001.npy ff_txt/data/test/
cp joint_txt/joint_reps/joint_txt_LAST/validation/joint_txt_hidden-00001-of-00001.npy ff_txt/data/validation/
python norm_joint.py
cp feature/train_lab_data.npy ff_img/data
cp feature/test_lab_data.npy ff_img/data
cp feature/validation_lab_data.npy ff_img/data
cp feature/train_lab_data.npy ff_txt/data
cp feature/test_lab_data.npy ff_txt/data
cp feature/validation_lab_data.npy ff_txt/data

echo "run ff for image"
sh ff_img/runall.sh
echo "run ff for text"
sh ff_txt/runall.sh

echo "BAE : iteration 1"
python bae_lab/gen_bae_data.py
sh bae_lab/runall_bae.sh
echo "BAE : iteration 2"
python bae_rep/gen_bae_data.py
sh bae_rep/runall_bae.sh
echo "BAE : iteration 3"
python bae_rep2/gen_bae_data.py
sh bae_rep2/runall_bae.sh
echo "BAE : iteration 4"
python bae_rep3/gen_bae_data.py
sh bae_rep3/runall_bae.sh

python calMap.py train

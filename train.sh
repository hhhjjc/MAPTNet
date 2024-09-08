#! /bin/sh

export CUDA_VISIBLE_DEVICES='0'
python train.py --config='config/SSD/fold0_resnet50.yaml'   2>&1 | tee ./exp/SSD/fold0_resnet50/model_1shot/output.log

# export CUDA_VISIBLE_DEVICES='0'
# python train.py --config='config/SSD/fold0_resnet50.yaml'   2>&1 | tee ./exp/SSD/fold1_resnet50/model_1shot/output.log

# export CUDA_VISIBLE_DEVICES='1'
# python train.py --config='config/SSD/fold0_resnet50.yaml'   2>&1 | tee ./exp/SSD/fold2_resnet50/model_1shot/output.log

######################################



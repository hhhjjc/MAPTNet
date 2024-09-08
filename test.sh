#! /bin/sh

export CUDA_VISIBLE_DEVICES='0'
python test.py --config='config/SSD/fold0_resnet50_test.yaml'   2>&1 | tee ./result/save_0/output.log
#!/bin/bash


# 모델명을 받구
# 무엇을 할지?


export PYTHONPATH=$PYTHONPATH:/home/xerato/workspace/tmp/models:/home/xerato/workspace/tmp/models/research:/home/xerato/workspace/tmp/models/research/slim


python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/190310_imgdata_25 --output_path=train.record

python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_pets.config


python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix training/model.ckpt-20000 --output_directory inference_graph

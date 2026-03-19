# run_start
python spatial_actor/train.py \
--device 2,3,4,5 \
--iter-based \
--data-folder /data/et24-hgf/train \
--train-replay-dir /data/et24-hgf/replay/replay_train \
--log-dir run_logs \
--cfg_path spatial_actor/configs/spact.yaml \
--cfg_opts "exp_id 6_tasks_froze_resnet bs 12"

### test
CUDA_VISIBLE_DEVICES=5 \
python spatial_actor/eval.py \
--eval-datafolder /data/et24-hgf/test \
--model-path /home/et24-huanggf/robot/SpatialActor/run_logs/test_EXP_test/model_50.pth \
--tasks all \
--device 0 \
--eval-episodes 25 \
--log-name rlbench_all \
--headless

### test_6_tasks
CUDA_VISIBLE_DEVICES=1 \
python spatial_actor/eval.py \
--eval-datafolder /data/et24-hgf/test \
--model-path /home/et24-huanggf/robot/SpatialActor/run_logs/6_tasks_no_depthexpert_EXP_6_TSK_no_depthexpert/model_50.pth \
--tasks put_item_in_drawer,stack_blocks,place_shape_in_shape_sorter,open_drawer,push_buttons,stack_cups \
--device 0 \
--eval-episodes 25 \
--log-name rlbench_6_tasks \
--headless

# run_start
python spatial_actor/train.py \
--device 2,3,4,5 \
--iter-based \
--data-folder /data/et24-hgf/train \
--train-replay-dir /data/et24-hgf/replay/replay_train \
--log-dir run_logs \
--cfg_path spatial_actor/configs/spact.yaml \
--cfg_opts "exp_id 6_tasks_froze_resnet bs 12"

########################## test for all tasks
CUDA_VISIBLE_DEVICES=0 \
python spatial_actor/eval.py \
--eval-datafolder /data/et24-hgf/test \
--model-path /home/et24-huanggf/robot/SpatialActor/run_logs/huashi_a100_6tasks_nocange/model_50.pth \
--tasks all \
--device 0 \
--eval-episodes 25 \
--log-name rlbench_all \
--headless

####################### test_6_tasks
CUDA_VISIBLE_DEVICES=0 \
python spatial_actor/eval.py \
--eval-datafolder /data/et24-hgf/test \
--model-path /home/et24-huanggf/robot/SpatialActor/run_logs/huashi_a100_6tasks_nocange/model_50.pth \
--tasks put_item_in_drawer,stack_blocks,place_shape_in_shape_sorter,open_drawer,push_buttons,stack_cups \
--device 0 \
--eval-episodes 25 \
--log-name rlbench_6_tasks \
--headless

##### clip RN50 + RN18 + 改融合机制
python spatial_actor/train.py \
--device 3 \
--iter-based \
--data-folder /data/et24-hgf/train \
--train-replay-dir /data/et24-hgf/replay/replay_train \
--log-dir run_logs \
--cfg_path spatial_actor/configs/spact.yaml \
--cfg_opts "exp_id 6_tasks_clipRN50_newconcat bs 8"

### clip50+concat+
python spatial_actor/train.py \
--device 3 \
--iter-based \
--data-folder /data/et24-hgf/train \
--train-replay-dir /data/et24-hgf/replay/replay_train \
--log-dir run_logs \
--cfg_path spatial_actor/configs/spact.yaml \
--cfg_opts "exp_id clip50+cross_attention bs 8 tasks put_item_in_drawer"

### new-attention+置信度预测
source /home/et24-huanggf/anaconda3/etc/profile.d/conda.sh
conda activate spact
cd /home/et24-huanggf/robot/SpatialActor
python spatial_actor/train.py --cfg_path spatial_actor/configs/spact_ablation_sparse.yaml --device 3

################# patchify + 解耦 训练代码
python spatial_actor/train.py \
--device 4,5 \
--iter-based \
--data-folder /data/et24-hgf/train \
--train-replay-dir /data/et24-hgf/replay/replay_train \
--log-dir run_logs \
--cfg_path spatial_actor/configs/spact.yaml \
--cfg_opts "exp_id tasks6_patchify+decouple bs 8"
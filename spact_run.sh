# run_start
python spatial_actor/train.py \
--device 0,1,2,3,4 \
--iter-based \
--data-folder /data/et24-hgf/train \
--train-replay-dir /data/et24-hgf/replay/replay_train \
--log-dir run_logs \
--cfg_path spatial_actor/configs/spact.yaml \
--cfg_opts "exp_id 5_train_vits"

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

CUDA_VISIBLE_DEVICES=0 \
python spatial_actor/eval.py \
--eval-datafolder /data/et24-hgf/test \
--model-path /home/et24-huanggf/robot/SpatialActor/run_logs/test_EXP_test/model_20.pth \
--tasks all \
--device 0 \
--eval-episodes 25 \
--log-name rlbench_all_2_with_original_eval_20 \
--headless
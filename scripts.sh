python -m hifigan.train --audio_root_path ./dataset_raw \
--feature_root_path ./dataset \
--input_training_file data_splits/wavlm-hifigan-train.csv \
--input_validation_file data_splits/wavlm-hifigan-valid.csv \
--checkpoint_path ./checkpoints \
--fp16 False --config hifigan/config_v1_wavlm.json \
--stdout_interval 25 \
--training_epochs 1800 --fine_tuning \
--checkpoint_interval 1000
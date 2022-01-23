python ./main.py \
    --model_config_path ./config/train_pems04_speed.yaml \
    --train_config_path ./config/train_config.yaml \
    --model_name AdapGLD \
    --num_epoch 10 \
    --num_iter 15 \
    --model_save_path ./model_states/AdapGLD_pems04_speed.pkl \
    --max_graph_num 3
python ./main.py \
    --model_config_path ./config/model_config_pems04.yaml \
    --train_config_path ./config/train_config.yaml \
    --model_name AdapGLD \
    --num_epoch 10 \
    --num_iter 15 \
    --model_save_path ./model_states/AdapGLD_pems04.pkl \
    --max_graph_num 3
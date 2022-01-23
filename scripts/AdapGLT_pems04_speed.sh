python ./main.py \
    --model_config_path ./config/train_pems04_speed.yaml \
    --train_config_path ./config/train_config.yaml \
    --model_name AdapGLT \
    --num_epoch 5 \
    --num_iter 20 \
    --model_save_path ./model_states/AdapGLT_pems04_speed.pkl \
    --max_graph_num 3
# AdapGL

This is a Pytorch implemention of AdapGL.

## Requirements

The model is implemented using python3 with dependencies specified in requirements.txt.

## Traffic datasets

PeMSD4 and PeMSD8 datasets can be downloaded from [PeMS-BAY](https://pan.baidu.com/s/1divakE-NIAUzk8xKP6lngQ) with password `"qhoa"`. Move them into `data` folder.  

## Model Training (for PeMSD4)

- AdapGL+ASTGCN

```
python ./main.py \
    --model_config_path ./config/model_config_pems04.yaml \
    --train_config_path ./config/train_config.yaml \
    --model_name AdapGLA \
    --num_epoch 5 \
    --num_iter 20 \
    --model_save_path ./model_states/AdapGLA_pems04.pkl \
    --max_graph_num 3
```
- AdapGL+DCRNN
```
python ./main.py \
    --model_config_path ./config/model_config_pems04.yaml \
    --train_config_path ./config/train_config.yaml \
    --model_name AdapGLD \
    --num_epoch 10 \
    --num_iter 15 \
    --model_save_path ./model_states/AdapGLD_pems04.pkl \
    --max_graph_num 3
```
- AdapGL+TGCN
```
python main.py \
    --model_config_path ./config/model_config_pems04.yaml \
    --train_config_path ./config/train_config.yaml \
    --model_name AdapGLT \
    --num_epoch 5 \
    --num_iter 20 \
    --model_save_path ./model_states/AdapGLT_pems04.pkl \
    --max_graph_num 3
```

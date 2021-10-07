import os
import sys
import yaml
import torch
import argparse
import trainer
from utils import scaler
from models import AdapGL
from dataset import TPDataset
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser()
parser.add_argument('--model_config_path', type=str, default='./config/model_config_pems04.yaml',
                    help='Config path of models')
parser.add_argument('--train_config_path', type=str, default='./config/train_config.yaml',
                    help='Config path of Trainer')
parser.add_argument('--model_name', type=str, default='AdapGLT', help='Model name to train')
parser.add_argument('--num_epoch', type=int, default=5, help='Training times per epoch')
parser.add_argument('--num_iter', type=int, default=100, help='Maximum value for iteration')
parser.add_argument('--model_save_path', type=str, default='./model_states/AdapGLT_pems04.pkl',
                    help='Model save path')
parser.add_argument('--max_graph_num', type=int, default=3, help='Volume of adjacency matrix set')
args = parser.parse_args()


def load_config(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main():
    model_config = load_config(args.model_config_path)
    train_config = load_config(args.train_config_path)
    torch.manual_seed(train_config['seed'])
    torch.cuda.manual_seed(train_config['seed'])
    # ----------------------- Load data ------------------------
    Scaler = getattr(sys.modules['utils.scaler'], train_config['scaler'])
    data_scaler = Scaler(axis=(0, 1, 2))

    data_config = model_config['dataset']
    device = torch.device(data_config['device'])
    data_names = ('train.npz', 'valid.npz', 'test.npz')

    data_loaders = []
    for data_name in data_names:
        dataset = TPDataset(os.path.join(data_config['data_dir'], data_name))
        if data_name == 'train.npz':
            data_scaler.fit(dataset.data['x'])
        dataset.transform(data_scaler)
        data_loader = DataLoader(dataset, batch_size=data_config['batch_size'])
        data_loaders.append(data_loader)

    # --------------------- Trainer setting --------------------
    net_name = args.model_name
    net_config = model_config[net_name]
    net_config.update(data_config)

    Model = getattr(AdapGL, net_name, None)
    if Model is None:
        raise ValueError('Model {} is not right!'.format(net_name))
    net_pred = Model(**net_config).to(device)
    net_graph = AdapGL.GraphLearn(
        net_config['num_nodes'],
        net_config['init_feature_num'],
    ).to(device)

    Optimizer = getattr(sys.modules['torch.optim'], train_config['optimizer'])
    optimizer_pred = Optimizer(
        net_pred.parameters(),
        lr=train_config['learning_rate'],
        weight_decay=train_config['weight_decay']
    )
    optimizer_graph = Optimizer(net_graph.parameters(), lr=train_config['learning_rate'])

    sc = train_config.get('lr_scheduler', None)
    if sc is None:
        scheduler_pred, scheduler_graph = None, None
    else:
        sc_name = sc.pop('name')
        Scheduler = getattr(sys.modules['torch.optim.lr_scheduler'], sc_name)
        scheduler_pred = Scheduler(optimizer_pred, **sc)
        scheduler_graph = None

    # --------------------------- Train -------------------------
    net_trainer = trainer.AdapGLTrainer(
        net_config['adj_mx_path'], net_pred, net_graph, optimizer_pred, optimizer_graph,
        scheduler_pred, scheduler_graph, args.num_epoch, args.num_iter,
        args.max_graph_num, data_scaler, args.model_save_path
    )
    # net_trainer = trainer.AdapGLE2ETrainer(
    #     net_config['adj_mx_path'], net_pred, net_graph, optimizer_pred, optimizer_graph,
    #     scheduler_pred, scheduler_graph, args.num_epoch, args.num_iter, data_scaler, args.model_save_path
    # )

    net_trainer.train(data_loaders[0], data_loaders[1])
    net_trainer.test(data_loaders[-1])


if __name__ == '__main__':
    main()

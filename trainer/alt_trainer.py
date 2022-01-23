import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from shutil import copyfile
from models.adj_mx import get_adj
from .base import Trainer, TFTrainer


class AdapGLTrainer(Trainer):
    """
    Adaptive Graph Learning Networks Trainer.

    Args:
        adj_mx_path: Paths of all adjacent matrixes which are splited by ','.
        model_pred: Model for prediction.
        model_graph: Model for graph learning.
        optimizer_pred: Optimizer for prediction model training.
        optimizer_graph: Optimizer for graph learning model training.
        scheduler_pred: Learning rate scheduler for prdiction model training.
        scheduler_graph: Learning rate scheduler for graph learning model training.
        epoch_num: Training epoch for prediction model and graph learning model per iteration.
        num_iter: Number of iteration.
        max_adj_num: The maximal volume of adj_mx set.
        scaler: Scaler for data set.
        model_save_path: Path to save of prediction model.
    """
    def __init__(self, adj_mx_path, model_pred, model_graph, optimizer_pred, optimizer_graph, scheduler_pred,
                 scheduler_graph, epoch_num, num_iter, max_adj_num, scaler, model_save_path):
        self.model_pred: nn.Module = model_pred
        self.model_graph: nn.Module = model_graph
        self.num_iter: int = num_iter
        self.model_save_path: str = model_save_path
        self.device = next(self.model_pred.parameters()).device

        self.max_adj_num: int = max_adj_num
        adj_mx_list = self.__get_adj_mx_list(adj_mx_path.split(','))
        self.adj_mx_list = [[adj_mx, -1] for adj_mx in adj_mx_list]
        self.epsilon = 1 / adj_mx_list[0].size(0) * 0.5

        self.best_adj_mx = None
        self.update_best_adj_mx('union', threshold=False)

        model_save_dir, model_name = os.path.split(self.model_save_path)
        self.graph_save_path = os.path.join(model_save_dir, 'GRAPH.pkl')

        if not os.path.exists(model_save_dir):
            os.mkdir(model_save_dir)

        self.model_pred_trainer = self.ModelPredTrainer(
            model_pred, optimizer_pred, scheduler_pred, epoch_num, scaler,
            model_save_path, self)

        self.model_graph_trainer = self.GraphLearnTrainer(
            model_graph, optimizer_graph, scheduler_graph, epoch_num, scaler,
            self.graph_save_path, self)

        # my own variables.
        best_save_dir = os.path.join(model_save_dir, model_name.split('.')[0])
        self.best_pred_path = os.path.join(best_save_dir, model_name)
        self.best_graph_path = os.path.join(best_save_dir, 'best_adj_mx.npy')

        if not os.path.exists(best_save_dir):
            os.mkdir(best_save_dir)

    def __get_adj_mx_list(self, adj_path_list):
        adj_mx_list = []
        for adj_path in adj_path_list:
            adj_mx = get_adj(np.load(adj_path.strip()), 'gcn')
            adj_mx = torch.tensor(adj_mx, dtype=torch.float32, device=self.device)
            adj_mx_list.append(adj_mx)
        return adj_mx_list

    def update_adj_mx_list(self, data_loader, new_adj_mx):
        # update self.adj_mx_list
        self.adj_mx_list.append([new_adj_mx, 0])
        max_loss, max_index = torch.finfo(torch.float32).min, -1

        for i, (adj_mx, _) in enumerate(self.adj_mx_list):
            cur_loss = self.evaluate(data_loader, adj_mx)
            self.adj_mx_list[i][-1] = cur_loss
            if cur_loss > max_loss:
                max_loss, max_index = cur_loss, i

        if len(self.adj_mx_list) > self.max_adj_num:
            self.adj_mx_list.pop(max_index)

    def update_best_adj_mx(self, criteria, threshold=True):
        """
        Update self.best_adj_mx.

        criteria:
            - replace: use the newest subgraph as best_adj_mx;
            - union: combine the adj_mx in self.adj_mx_list;
            - weight_union: weighted sum of adj_mx in self.adj_mx_list according to
                evaluate loss.
        """
        if criteria == 'replace':
            best_adj_mx = self.adj_mx_list[-1][0]
        elif criteria == 'union':
            adj_mx_sum = torch.zeros_like(self.adj_mx_list[0][0])
            adj_num_sum = torch.zeros_like(adj_mx_sum)
            for adj_mx, _ in self.adj_mx_list:
                adj_mx_sum += adj_mx
                adj_num_sum += (1 + torch.sign(adj_mx - 1e-4)) / 2
            adj_mx_sum /= adj_num_sum
            adj_mx_sum[torch.logical_or(torch.isnan(adj_mx_sum), torch.isinf(adj_mx_sum))] = 0
            best_adj_mx = adj_mx_sum
        else:
            loss_tensor = torch.tensor([x[-1] for x in self.adj_mx_list], requires_grad=False)
            loss_weight = F.softmax(loss_tensor.max() - loss_tensor, dim=0)

            best_adj_mx = torch.zeros_like(self.adj_mx_list[0][0])
            for i, (adj_mx, _) in enumerate(self.adj_mx_list):
                best_adj_mx += loss_weight[i] * adj_mx
        if threshold:
            d = best_adj_mx.sum(dim=-1) ** (-0.5)
            best_adj_mx = torch.relu(d.view(-1, 1) * best_adj_mx * d - self.epsilon)
        d = best_adj_mx.sum(dim=-1) ** (-0.5)
        self.best_adj_mx = d.view(-1, 1) * best_adj_mx * d

    def update_num_epoch(self, cur_iter):
        if cur_iter == self.num_iter // 2 + 1:
            self.model_pred_trainer.max_epoch_num += 5

    def train_one_epoch(self, train_data_loader, eval_data_loader, metrics, cur_iter):
        self.model_pred_trainer.train(train_data_loader, eval_data_loader, metrics)
        self.model_pred.load_state_dict(torch.load(self.model_save_path))
        self.model_graph_trainer.train(train_data_loader, eval_data_loader, metrics)
        self.model_graph.load_state_dict(torch.load(self.graph_save_path))
        # Add the learned graph to self.adj_mx_list, update self.best_adj_mx.
        new_adj_mx = self.model_graph(self.best_adj_mx).detach()
        print('Evaluation results of all subgraphs:')
        self.update_adj_mx_list(eval_data_loader, new_adj_mx)
        if cur_iter > self.num_iter * 0.8:
            self.update_best_adj_mx('replace')
        else:
            self.update_best_adj_mx('weight_union')

    @torch.no_grad()
    def evaluate(self, data_loader, adj_mx):
        """
        Test the prediction loss on data set 'data_loader' using 'adj_mx'.
        """
        loss, _, _ = self.model_pred_trainer.evaluate(data_loader, adj_mx=adj_mx)
        return loss

    @torch.no_grad()
    def test(self, data_loader, metrics=('mae', 'rmse', 'mape')):
        self.model_pred.load_state_dict(torch.load(self.best_pred_path))
        best_adj_mx_np = np.load(self.best_graph_path)
        best_adj_mx = torch.tensor(
            data=best_adj_mx_np,
            dtype=torch.float32,
            device=self.device
        )
        sparsity = (best_adj_mx_np == 0).sum() / (best_adj_mx_np.shape[0] ** 2)
        print('Sparsity: {:.4f}'.format(sparsity))
        print('Test results of current graph: ')
        _, y_true, y_pred = self.model_pred_trainer.evaluate(data_loader, metrics, adj_mx=best_adj_mx)
        self.model_pred_trainer.print_test_result(y_pred, y_true, metrics)

    def train(self, train_data_loader, eval_data_loader, metrics=('mae', 'rmse', 'mape')):
        print('Start Training...')
        min_loss = torch.finfo(torch.float32).max
        for i in range(self.num_iter):
            print('Iteration {}:'.format(i + 1))
            self.train_one_epoch(train_data_loader, eval_data_loader, metrics, i)
            print('Evaluation results of current graph:')
            cur_loss = self.evaluate(eval_data_loader, self.best_adj_mx)
            if cur_loss < min_loss:
                copyfile(self.model_save_path, self.best_pred_path)
                np.save(self.best_graph_path, self.best_adj_mx.cpu().numpy())
                min_loss = cur_loss
            self.update_num_epoch(i + 1)

    class ModelPredTrainer(TFTrainer):
        def __init__(self, model, optimizer, lr_scheduler, max_epoch_num, scaler, model_save_path, outer_obj):
            super().__init__(model, optimizer, lr_scheduler, max_epoch_num, scaler, model_save_path)
            self.outer_obj: AdapGLTrainer = outer_obj
            self.num_iter: int = 1
            self.batches_seen: int = 0

        def train_one_epoch(self, data_loader):
            self.model.train()
            adj_mx = self.outer_obj.best_adj_mx
            for _ in range(self.num_iter):
                for x, y in data_loader:
                    x = x.type(torch.float32).to(self.device)
                    y = y.type(torch.float32).to(self.device)
                    if str(self.model) != 'AdapGLD':
                        pred = self.model(x, adj_mx)
                    else:
                        pred = self.model(x, adj_mx, labels=y, batches_seen=self.batches_seen)
                    loss = self.model_loss_func(pred, y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.batches_seen += 1

        def train(self, train_data_loader, eval_data_loader, metrics=('mae', 'rmse', 'mape')):
            print('Round for prediction model:')
            super().train(train_data_loader, eval_data_loader, metrics)

        @torch.no_grad()
        def evaluate(self, data_loader, metrics=('mae', 'rmse', 'mape'), adj_mx=None):
            if adj_mx is None:
                adj_mx = self.outer_obj.best_adj_mx
            return super().evaluate(data_loader, metrics, adj_mx=adj_mx)

        @torch.no_grad()
        def test(self, data_loader, metrics=('mae', 'rmse', 'mape')):
            self.model.load_state_dict(torch.load(self.model_save_path))
            _, y_true, y_pred = self.evaluate(data_loader, metrics)
            self.print_test_result(y_pred, y_true, metrics)

        @staticmethod
        def model_loss_func(y_pred, y_true):
            return F.l1_loss(y_pred, y_true)

    class GraphLearnTrainer(TFTrainer):
        def __init__(self, model, optimizer, lr_scheduler, max_epoch_num, scaler, model_save_path, outer_obj):
            super().__init__(model, optimizer, lr_scheduler, max_epoch_num, scaler, model_save_path)
            self.outer_obj: AdapGLTrainer = outer_obj
            self.num_iter: int = 1
            self.delta: int = 0.05

        def train_one_epoch(self, data_loader):
            self.model.train()
            best_adj_mx = self.outer_obj.best_adj_mx
            for _ in range(self.num_iter):
                for x, y in data_loader:
                    x = x.type(torch.float32).to(self.device)
                    y = y.type(torch.float32).to(self.device)
                    adj_mx = self.model(best_adj_mx)
                    pred = self.outer_obj.model_pred(x, adj_mx)
                    loss = self.model_loss_func(pred, y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

        def train(self, train_data_loader, eval_data_loader, metrics=('mae', 'rmse', 'mape')):
            print('Round for graph learning:')
            super().train(train_data_loader, eval_data_loader, metrics)

        @torch.no_grad()
        def evaluate(self, data_loader, metrics=('mae', 'rmse', 'mape')):
            adj_mx = self.model(self.outer_obj.best_adj_mx).detach()
            return self.outer_obj.model_pred_trainer.evaluate(data_loader, metrics, adj_mx=adj_mx)

        @torch.no_grad()
        def test(self, data_loader, metrics=('mae', 'rmse', 'mape')):
            self.model.load_state_dict(torch.load(self.model_save_path))
            _, y_true, y_pred = self.evaluate(data_loader, metrics)
            self.print_test_result(y_pred, y_true, metrics)

        def model_loss_func(self, y_pred, y_true):
            """Loss function of Graph Learn Model."""
            mx_p = self.outer_obj.best_adj_mx
            mx_q = self.model(mx_p)
            mx_delta = torch.sign(mx_q) - torch.sign(mx_p)
            sim_loss = F.relu(F.relu(mx_delta).mean() - self.delta) / self.delta
            pred_loss = F.l1_loss(y_pred, y_true)
            return pred_loss + sim_loss

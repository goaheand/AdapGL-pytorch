import os
import sys
import abc
from typing import List
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from shutil import copyfile
from models.adj_mx import get_adj
from utils.metrics import get_mae, get_mape, get_rmse
from utils.func_util import time_decorator
from utils.train_tool import EarlyStop


class Trainer(metaclass=abc.ABCMeta):
    @staticmethod
    def get_eval_result(y_pred, y_true, metrics=('mae', 'rmse', 'mape')):
        module = sys.modules[__name__]

        eval_results = []
        for metric_name in metrics:
            eval_func = getattr(module, 'get_{}'.format(metric_name))
            eval_results.append(eval_func(y_pred, y_true))

        return eval_results

    @abc.abstractmethod
    def train(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def train_one_epoch(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def evaluate(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def test(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def model_loss_func(self, y_pred, y_true, *args):
        pass


class GeneralTrainer(Trainer):
    def __init__(self, model, optimizer, lr_scheduler, max_epoch_num, scaler, model_save_path):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.max_epoch_num = max_epoch_num
        self.scaler = scaler
        self.model_save_path = model_save_path
        self.model_save_dir = os.path.dirname(model_save_path)
        self.early_stop = EarlyStop(5, min_is_best=True)
        self.device = next(self.model.parameters()).device

        if not os.path.exists(self.model_save_dir):
            os.mkdir(self.model_save_dir)

    @time_decorator
    def train(self, train_data_loader, eval_data_loader, metrics=('mae', 'rmse', 'mape')):
        tmp_state_save_path = os.path.join(self.model_save_dir, 'temp.pkl')
        min_loss = torch.finfo(torch.float32).max

        for epoch in range(1, self.max_epoch_num + 1):
            # train one epoch
            self.train_one_epoch(train_data_loader)

            # evaluate
            print('Epoch {}'.format(epoch), end='  ')
            eval_loss, _, _ = self.evaluate(eval_data_loader, metrics)

            # Criteria for early stopping
            if self.early_stop.reach_stop_criteria(eval_loss):
                break

            # save model state when meeting minimum loss
            # save to a temporary path first to avoid overwriting original state.
            if eval_loss < min_loss:
                torch.save(self.model.state_dict(), tmp_state_save_path)
                min_loss = eval_loss

            # learning rate decay
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        self.early_stop.reset()
        copyfile(tmp_state_save_path, self.model_save_path)
        os.remove(tmp_state_save_path)

    @torch.no_grad()
    def evaluate(self, data_loader, metrics, **kwargs):
        self.model.eval()

        y_true, y_pred, loss, batch_num = [], [], 0, 0
        for x, y in data_loader:
            x = x.type(torch.float32).to(self.device)
            y = y.type(torch.float32).to(self.device)
            pred = self.model(x, **kwargs).detach()
            loss += self.model_loss_func(pred, y).item()
            batch_num += 1
            y_true.append(y)
            y_pred.append(pred)
        y_true = self.scaler.inverse_transform(torch.cat(y_true, dim=0).cpu().numpy(), axis=0)
        y_pred = self.scaler.inverse_transform(torch.cat(y_pred, dim=0).cpu().numpy(), axis=0)

        eval_results = self.get_eval_result(y_pred, y_true, metrics)
        for metric_name, eval_ret in zip(metrics, eval_results):
            print('{}:  {:.4f}'.format(metric_name.upper(), eval_ret), end='  ')
        print()

        return loss / batch_num, y_true, y_pred

    def print_test_result(self, y_pred, y_true, metrics):
        for i in range(y_true.shape[1]):
            metric_results = self.get_eval_result(y_pred[:, i], y_true[:, i], metrics)
            print('Horizon {}'.format(i + 1), end='  ')
            for j in range(len(metrics)):
                print('{}:  {:.4f}'.format(metrics[j], metric_results[j]), end='  ')
            print()


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
    def __init__(self, adj_mx_path, model_pred, model_graph, optimizer_pred, optimizer_graph,
                 scheduler_pred, scheduler_graph, epoch_num, num_iter, max_adj_num, scaler,
                 model_save_path):
        self.model_pred = model_pred
        self.model_graph = model_graph
        self.optimizer_pred = optimizer_pred
        self.optimizer_graph = optimizer_graph
        self.scheduler_pred = scheduler_pred
        self.scheduler_graph = scheduler_graph
        self.epoch_num = epoch_num
        self.num_iter = num_iter
        self.scaler = scaler
        self.model_save_path = model_save_path
        self.device = next(self.model_pred.parameters()).device

        self.max_adj_num = max_adj_num
        adj_mx_list = self._get_adj_mx_list(adj_mx_path.split(','))
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

    def _get_adj_mx_list(self, adj_path_list):
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
            adj_mx_sum[torch.isnan(adj_mx_sum)] = 0
            adj_mx_sum[torch.isinf(adj_mx_sum)] = 0
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

    @staticmethod
    def model_loss_func(y_pred, y_true):
        return -1

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

    class ModelPredTrainer(GeneralTrainer):
        def __init__(self, model, optimizer, lr_scheduler, max_epoch_num, scaler,
                     model_save_path, outer_obj):
            super().__init__(model, optimizer, lr_scheduler, max_epoch_num, scaler, model_save_path)
            self.outer_obj: AdapGLTrainer = outer_obj
            self.num_iter = 1
            self.batches_seen = 0

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

    class GraphLearnTrainer(GeneralTrainer):
        def __init__(self, model, optimizer, lr_scheduler, max_epoch_num, scaler,
                     model_save_path, outer_obj):
            super().__init__(model, optimizer, lr_scheduler, max_epoch_num, scaler, model_save_path)
            self.outer_obj: AdapGLTrainer = outer_obj
            self.num_iter = 1
            self.delta = 0.05

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


class AdapGLE2ETrainer(GeneralTrainer):
    def __init__(self, adj_mx_path, model_pred, model_graph, optimizer_pred, optimizer_graph,
                 scheduler_pred, scheduler_graph, epoch_num, num_iter, scaler, model_save_path):
        super().__init__(model_pred, optimizer_pred, scheduler_pred, epoch_num, scaler, model_save_path)

        self.model_graph = model_graph
        self.optimizer_graph = optimizer_graph
        self.scheduler_graph = scheduler_graph
        self.num_iter = num_iter

        self.device = next(self.model.parameters()).device
        self._delta = 0.25

        adj_mx_list = self._get_adj_mx_list(adj_mx_path.split(','))
        self.cur_adj_mx = self.update_best_adj_mx(adj_mx_list)
        self.batches_seen = 0

        self._model_save_dir = os.path.dirname(self.model_save_path)
        self._graph_save_path = os.path.join(self._model_save_dir, 'best_adj_mx.npy')
        self.early_stop = EarlyStop(10, min_is_best=True)
    
    def train_one_epoch(self, data_loader):
        self.model.train()
        self.model_graph.train()

        for x, y in data_loader:
            x = x.type(torch.float32).to(self.device)
            y = y.type(torch.float32).to(self.device)
            adj_mx = self.model_graph(self.cur_adj_mx)
            if str(self.model) != 'AdapGLD':
                pred = self.model(x, adj_mx)
            else:
                pred = self.model(x, adj_mx, labels=y, batches_seen=self.batches_seen)
            loss = self.model_loss_func(pred, y, adj_mx)
            self.optimizer.zero_grad()
            self.optimizer_graph.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.optimizer_graph.step()
            self.batches_seen += 1
        
        return self.model_graph(self.cur_adj_mx).detach()

    @time_decorator
    def train(self, train_data_loader, eval_data_loader, metrics=('mae', 'rmse', 'mape')):
        tmp_state_save_path = os.path.join(self._model_save_dir, 'temp.pkl')
        min_loss = torch.finfo(torch.float32).max

        for epoch in range(1, self.max_epoch_num + 1):
            # train one epoch
            adj_mx = self.train_one_epoch(train_data_loader)

            # evaluate
            print('Epoch {}'.format(epoch), end='  ')
            eval_loss, _, _ = self.evaluate(eval_data_loader, metrics, adj_mx=adj_mx)

            # Criteria for early stopping
            if self.early_stop.reach_stop_criteria(eval_loss):
                break

            # save model state when meeting minimum loss
            # save to a temporary path first to avoid overwriting original state.
            if eval_loss < min_loss:
                torch.save(self.model.state_dict(), tmp_state_save_path)
                np.save(self._graph_save_path, adj_mx.cpu().numpy())
                min_loss = eval_loss

            # learning rate decay
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            if self.scheduler_graph is not None:
                self.scheduler_graph.step()

        self.early_stop.reset()
        copyfile(tmp_state_save_path, self.model_save_path)
        os.remove(tmp_state_save_path)
    
    def test(self, data_loader, metrics=('mae', 'rmse', 'mape')):
        self.model.load_state_dict(torch.load(self.model_save_path))
        adj_mx = torch.tensor(
            data=np.load(self._graph_save_path),
            dtype=torch.float32,
            device=self.device
        )
        self.evaluate(data_loader, metrics, adj_mx=adj_mx)
    
    def model_loss_func(self, y_pred, y_true, cur_adj_mx=None):
        pred_loss = F.l1_loss(y_pred, y_true)
        if self.model.training:
            mx_p, mx_q = self.cur_adj_mx, cur_adj_mx
            mx_delta = torch.sign(mx_q) - torch.sign(mx_p)
            sim_loss = F.relu(F.relu(mx_delta).mean() - self._delta) / self._delta
            pred_loss += sim_loss
        return pred_loss

    def _get_adj_mx_list(self, adj_path_list):
        adj_mx_list = []
        for adj_path in adj_path_list:
            adj_mx = get_adj(np.load(adj_path.strip()), 'gcn')
            adj_mx = torch.tensor(adj_mx, dtype=torch.float32, device=self.device)
            adj_mx_list.append(adj_mx)
        return adj_mx_list
    
    @staticmethod
    def update_best_adj_mx(adj_mx_list: List):
        adj_mx_sum = torch.zeros_like(adj_mx_list[0])
        adj_num_sum = torch.zeros_like(adj_mx_sum)
        for adj_mx in adj_mx_list:
            adj_mx_sum += adj_mx
            adj_num_sum += (1 + torch.sign(adj_mx - 1e-4)) / 2
        adj_mx_sum /= adj_num_sum
        adj_mx_sum[torch.isnan(adj_mx_sum)] = 0
        adj_mx_sum[torch.isinf(adj_mx_sum)] = 0

        best_adj_mx = adj_mx_sum
        d = best_adj_mx.sum(dim=-1) ** (-0.5)
        best_adj_mx = d.view(-1, 1) * best_adj_mx * d
        return best_adj_mx

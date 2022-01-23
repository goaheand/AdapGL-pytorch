import os
import torch
import numpy as np
import torch.nn.functional as F
from shutil import copyfile
from .base import TFTrainer
from models.adj_mx import get_adj
from utils.train_tool import EarlyStop, time_decorator


class AdapGLE2ETrainer(TFTrainer):
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
    def update_best_adj_mx(adj_mx_list):
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

import os
import sys
import torch
from shutil import copyfile
from utils.train_tool import EarlyStop, time_decorator
from utils.metrics import get_mae, get_mape, get_rmse


class Trainer:
    @staticmethod
    def get_eval_result(y_pred, y_true, metrics=('mae', 'rmse', 'mape')):
        module = sys.modules[__name__]

        eval_results = []
        for metric_name in metrics:
            eval_func = getattr(module, 'get_{}'.format(metric_name))
            eval_results.append(eval_func(y_pred, y_true))

        return eval_results

    def train(self, *args, **kwargs):
        raise NotImplementedError

    def train_one_epoch(self, *args, **kwargs):
        raise NotImplementedError

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError

    def model_loss_func(self, y_pred, y_true, *args):
        return 0


class TFTrainer(Trainer):
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

        y_true, y_pred, loss, data_num = [], [], 0, 0
        for x, y in data_loader:
            x = x.type(torch.float32).to(self.device)
            y = y.type(torch.float32).to(self.device)
            pred = self.model(x, **kwargs).detach()
            loss += self.model_loss_func(pred, y).item()
            data_num += len(data_loader)
            y_true.append(y)
            y_pred.append(pred)
        y_true = self.scaler.inverse_transform(torch.cat(y_true, dim=0).cpu().numpy(), axis=0)
        y_pred = self.scaler.inverse_transform(torch.cat(y_pred, dim=0).cpu().numpy(), axis=0)

        eval_results = self.get_eval_result(y_pred, y_true, metrics)
        for metric_name, eval_ret in zip(metrics, eval_results):
            print('{}:  {:.4f}'.format(metric_name.upper(), eval_ret), end='  ')
        print()

        return loss / data_num, y_true, y_pred

    def print_test_result(self, y_pred, y_true, metrics):
        for i in range(y_true.shape[1]):
            metric_results = self.get_eval_result(y_pred[:, i], y_true[:, i], metrics)
            print('Horizon {}'.format(i + 1), end='  ')
            for j in range(len(metrics)):
                print('{}:  {:.4f}'.format(metrics[j], metric_results[j]), end='  ')
            print()

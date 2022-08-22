import sys
from abc import ABC, abstractmethod
from typing import Tuple, Dict, List, Optional
from collections import defaultdict

import numpy as np
import torch


class Trainer(ABC):
    """
    Abstract class for experiments with 2 abstract methods train_step and test_step.
    Note that both the train_step and test_step are receive batch and return a dictionary.
    More detail about those is in each method's docstring.

    :param model: nn.Module
    :param device: torch.device
    :param mode: str
    :param optimizer: str, optional
    :param scheduler: str, optional
    """

    def __init__(self, model, device, mode: str, optimizer: Optional[torch.optim.Optimizer] = None, scheduler=None):
        self.device = device
        self.loader = None
        self.mode = mode
        self.class_weight = None
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model = model
        self.log_train, self.log_test = defaultdict(list), defaultdict(list)
        self.loss_buffer, self.weight_buffer = torch.inf, defaultdict(list)

    def _to_cpu(self, *args) -> Tuple[torch.Tensor, ...] or torch.Tensor:
        """
        Cast given arguments to cpu.
        :param args: Single argument or variable-length sequence of arguments.
        :return: Single argument or variable-length sequence of arguments in cpu.
        """
        if len(args) == 1:
            a = args[0]
            if isinstance(a, torch.Tensor):
                return a.item() if 'item' in a.__dir__() else a.detach().cpu() if 'detach' in a.__dir__() else a
            else:
                return a
        else:
            return tuple([self._to_cpu(a) for a in args])

    def _to_device(self, *args) -> Tuple[torch.Tensor, ...] or torch.Tensor:
        """
        Cast given arguments to device.
        :param args: Single argument or variable-length sequence of arguments.
        :return: Single argument or variable-length sequence of arguments in device.
        """
        if len(args) == 1:
            return args[0].to(self.device) if 'to' in args[0].__dir__() else args[0]
        else:
            return tuple([a.to(self.device) if 'to' in a.__dir__() else a for a in args])

    def run(self, n_epoch: int, loader, valid_loader=None) -> None:
        """
        Run for a given loader.
        If valid_loader is not None, the model evaluates the data in valid_loader for every epoch.
        :param n_epoch:
        :param loader:
        :param valid_loader:
        :return:
        """
        self.loader = loader
        for e in range(n_epoch):
            print('Epoch %d' % (e + 1))
            self.run_epoch(loader)
            if valid_loader:
                self.toggle()
                self.run_epoch(valid_loader)
                self.toggle()
            cur_loss = np.mean(self.log_test['loss'] if valid_loader else self.log_train['loss']).item()
            self.check_better(cur_epoch=e + 1, cur_loss=cur_loss)
            if self.mode == 'train': self.scheduler.step()

    def run_epoch(self, loader) -> None:
        self.log_reset()
        for i, batch in enumerate(loader):
            self.run_batch(batch)
            self.print_log(cur_batch=i+1, num_batch=len(loader))

    def run_batch(self, batch: tuple) -> None:
        """
        Run for a batch (not epoch)
        """
        batch = self._to_device(batch)
        if self.mode == 'train':
            self.model.train()
            self.optimizer.zero_grad()
            step_dict = self.train_step(batch)
            step_dict['loss'].backward()
            self.optimizer.step()
        elif self.mode == 'test':
            self.model.eval()
            with torch.no_grad():
                step_dict = self.test_step(batch)
        else:
            raise ValueError(f'Unexpected mode {self.mode}.')

        self.logging(step_dict, n=len(batch[0]))

    @abstractmethod
    def train_step(self, batch: tuple) -> Dict[str, torch.Tensor]:
        """
        Unused dummy function yet exist to provide I/O format information.
        The batch is a part of the dataset in the dataloader fetched via  __getitem__ method.
        The dictionary consists of {'str': torch.tensor(value)}, and must include 'loss'.
        Note that this function work only in 'train' mode, hence back() method in nn.Module is necessary.
        """
        x, y, fn = batch
        outputs = self.model(x)
        loss = [0.0] * len(x)
        return {'loss': loss, 'mean': torch.mean(outputs)}

    @abstractmethod
    def test_step(self, batch: tuple) -> Dict[str, torch.Tensor]:
        """
        Unused dummy function yet exist to provide I/O format information.
        The batch is a part of the dataset in the dataloader fetched via  __getitem__ method.
        The dictionary consists of {'str': torch.tensor(value)}, and must include 'loss'.
        Note that this function work only in 'test' mode.
        """
        x, y, fn = batch
        outputs = self.model(x)
        loss = [0.0] * len(x)
        return {'loss': loss, 'mean': torch.mean(outputs)}

    def logging(self, step_dict: Dict[str, torch.Tensor], n: int) -> None:
        for k, v in step_dict.items():
            if 'output' in k.lower():
                v = v.tolist()
                n = 1
            if self.mode == 'train':
                self.log_train[k] += ([self._to_cpu(v)] * n)
            elif self.mode == 'test':
                self.log_test[k] += ([self._to_cpu(v)] * n)
            else:
                raise ValueError(f'Unexpected mode {self.mode}.')

    def toggle(self, mode: str = None) -> None:
        """
        Switching mode in instance.
        :param mode: str
        """
        if mode:
            self.mode = mode
        else:
            self.mode = 'test' if self.mode == 'train' else 'train'

    def check_better(self, cur_epoch: int, cur_loss: float) -> None:
        """
        Compare the current model and the in-buffer model.
        The criterion is loss.
        :param cur_epoch: int
        :param cur_loss: float
        """
        if cur_loss > self.loss_buffer: return
        self.weight_buffer['epoch'].append(cur_epoch)
        self.weight_buffer['weight'].append(self.model.state_dict())

    def log_reset(self) -> None:
        """
        Clear the log.
        """
        self.log_train.clear()
        self.log_test.clear()

    def print_log(self, cur_batch: int, num_batch: int) -> None:
        log = self.log_train if self.mode == 'train' else self.log_test
        avg_loss, LR = np.mean(log['loss']), self.optimizer.param_groups[0]['lr'] if self.optimizer else 0
        is_end = cur_batch == num_batch
        disp = None if is_end else sys.stderr
        end = '\n' if is_end else ''

        log_str = f'\r\t\tBatch: {cur_batch}/{num_batch}'
        for k, v in log.items():
            if 'output' in k: continue
            log_str += f'\t{k}: {np.mean(v):.6f}'
        log_str += f'\tLR: {LR:.3e}'
        print(log_str, end=end, file=disp)

    def save_model(self, filename: str) -> None:
        torch.save(self.model.state_dict(), filename)

    def apply_weights(self, filename: str = None, strict: bool = True) -> None:
        """
        Update model's weights.
        If filename is None, model updated with latest weight buffer in the instance.
        If not, the model is updated with weights loaded from the filename.
        :param filename: str
        :param strict: bool
        """
        try:
            if filename:
                weights = torch.load(filename, map_location=torch.device("cpu"))
            else:
                weights = self.weight_buffer['weight'][-1]
        except FileNotFoundError as e:
            print(f'Cannot find file {filename}', e)
        except IndexError as e:
            print(f'There is no weight in buffer of trainer.', e)
        finally:
            self.model.load_state_dict(weights, strict=strict)

    def inference(self, loader):
        self.toggle('test')
        self.run_epoch(loader)
        return self.log_test['output']

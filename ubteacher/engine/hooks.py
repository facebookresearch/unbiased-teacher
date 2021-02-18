# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.engine.hooks import HookBase
import detectron2.utils.comm as comm

import torch
import numpy as np
from contextlib import contextmanager


class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader, model_output, model_name=""):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader
        self._model_output = model_output
        self._model_name = model_name

    def _do_loss_eval(self):
        record_acc_dict = {}
        with inference_context(self._model), torch.no_grad():
            for _, inputs in enumerate(self._data_loader):
                record_dict = self._get_loss(inputs, self._model)
                # accumulate the losses
                for loss_type in record_dict.keys():
                    if loss_type not in record_acc_dict.keys():
                        record_acc_dict[loss_type] = record_dict[loss_type]
                    else:
                        record_acc_dict[loss_type] += record_dict[loss_type]
            # average
            for loss_type in record_acc_dict.keys():
                record_acc_dict[loss_type] = record_acc_dict[loss_type] / len(
                    self._data_loader
                )

            # divide loss and other metrics
            loss_acc_dict = {}
            for key in record_acc_dict.keys():
                if key[:4] == "loss":
                    loss_acc_dict[key] = record_acc_dict[key]

            # only output the results of major node
            if comm.is_main_process():
                total_losses_reduced = sum(loss for loss in loss_acc_dict.values())
                self.trainer.storage.put_scalar(
                    "val_total_loss_val" + self._model_name, total_losses_reduced
                )

                record_acc_dict = {
                    "val_" + k + self._model_name: record_acc_dict[k]
                    for k in record_acc_dict.keys()
                }

                if len(record_acc_dict) > 1:
                    self.trainer.storage.put_scalars(**record_acc_dict)

    def _get_loss(self, data, model):
        if self._model_output == "loss_only":
            record_dict = model(data)

        elif self._model_output == "loss_proposal":
            record_dict, _, _, _ = model(data, branch="val_loss", val_mode=True)

        elif self._model_output == "meanteacher":
            record_dict, _, _, _, _ = model(data)

        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in record_dict.items()
        }

        return metrics_dict

    def _write_losses(self, metrics_dict):
        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        comm.synchronize()
        all_metrics_dict = comm.gather(metrics_dict, dst=0)

        if comm.is_main_process():
            # average the rest metrics
            metrics_dict = {
                "val_" + k: np.mean([x[k] for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }
            total_losses_reduced = sum(loss for loss in metrics_dict.values())

            self.trainer.storage.put_scalar("val_total_loss_val", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.trainer.storage.put_scalars(**metrics_dict)

    def _detect_anomaly(self, losses, loss_dict):
        if not torch.isfinite(losses).all():
            raise FloatingPointError(
                "Loss became infinite or NaN at iteration={}!\nloss_dict = {}".format(
                    self.trainer.iter, loss_dict
                )
            )

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)

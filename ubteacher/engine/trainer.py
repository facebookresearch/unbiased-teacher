# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import time
import math
import logging
import torch
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.cuda.amp import autocast
from fvcore.nn.precise_bn import get_bn_modules
import numpy as np
from collections import OrderedDict

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultTrainer, SimpleTrainer, TrainerBase
from detectron2.engine.train_loop import AMPTrainer
from detectron2.utils.events import EventStorage
from detectron2.evaluation import COCOEvaluator, verify_results
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.engine import hooks
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances

from ubteacher.data.build import (
    build_detection_semisup_train_loader,
    build_detection_test_loader,
    build_detection_semisup_train_loader_two_crops,
)
from ubteacher.data.dataset_mapper import DatasetMapperTwoCropSeparate
from ubteacher.engine.hooks import LossEvalHook
from ubteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel
from ubteacher.checkpoint.detection_checkpoint import DetectionTSCheckpointer
from ubteacher.solver.build import build_lr_scheduler


# Supervised-only Trainer
class BaselineTrainer(DefaultTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        Use the custom checkpointer, which loads other backbone models
        with matching heuristics.
        """
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )

        TrainerBase.__init__(self)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    def train_loop(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step()
                    self.after_step()
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    def run_step(self):
        self._trainer.iter = self.iter

        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()

        data = next(self._trainer._data_loader_iter)
        data_time = time.perf_counter() - start

        record_dict, _, _, _ = self.model(data, branch="supervised")

        num_gt_bbox = 0.0
        for element in data:
            num_gt_bbox += len(element["instances"])
        num_gt_bbox = num_gt_bbox / len(data)
        record_dict["bbox_num/gt_bboxes"] = num_gt_bbox

        loss_dict = {}
        for key in record_dict.keys():
            if key[:4] == "loss" and key[-3:] != "val":
                loss_dict[key] = record_dict[key]

        losses = sum(loss_dict.values())

        metrics_dict = record_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_semisup_train_loader(cfg, mapper=None)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable
        """
        return build_detection_test_loader(cfg, dataset_name)

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                cfg.TEST.EVAL_PERIOD,
                self.model,
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

    def _write_metrics(self, metrics_dict: dict):
        """
        Args:
            metrics_dict (dict): dict of scalar metrics
        """
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)

            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }

            loss_dict = {}
            for key in metrics_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = metrics_dict[key]

            total_losses_reduced = sum(loss for loss in loss_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)


# Unbiased Teacher Trainer
class UBTeacherTrainer(DefaultTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        Use the custom checkpointer, which loads other backbone models
        with matching heuristics.
        """
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        data_loader = self.build_train_loader(cfg)

        # create an student model
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)

        # create an teacher model
        model_teacher = self.build_model(cfg)
        self.model_teacher = model_teacher

        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )

        TrainerBase.__init__(self)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )
        self.scheduler = self.build_lr_scheduler(cfg, optimizer)

        # Ensemble teacher and student model is for model saving and loading
        ensem_ts_model = EnsembleTSModel(model_teacher, model)

        self.checkpointer = DetectionTSCheckpointer(
            ensem_ts_model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")

        if cfg.TEST.EVALUATOR == "COCOeval":
            return COCOEvaluator(dataset_name, cfg, True, output_folder)
        else:
            raise ValueError("Unknown test evaluator.")

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapperTwoCropSeparate(cfg, True)
        return build_detection_semisup_train_loader_two_crops(cfg, mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)

    def train(self):
        self.train_loop(self.start_iter, self.max_iter)
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    def train_loop(self, start_iter: int, max_iter: int):
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()

                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step_full_semisup()
                    self.after_step()
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    # =====================================================
    # ================== Pseduo-labeling ==================
    # =====================================================
    def threshold_bbox(self, proposal_bbox_inst, thres=0.7, proposal_type="roih"):
        if proposal_type == "rpn":
            valid_map = proposal_bbox_inst.objectness_logits > thres

            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            # create box
            new_bbox_loc = proposal_bbox_inst.proposal_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.objectness_logits = proposal_bbox_inst.objectness_logits[
                valid_map
            ]
        elif proposal_type == "roih":
            valid_map = proposal_bbox_inst.scores > thres

            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            # create box
            new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes[valid_map]
            new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map]

        return new_proposal_inst

    def process_pseudo_label(
        self, proposals_rpn_unsup_k, cur_threshold, proposal_type, psedo_label_method=""
    ):
        list_instances = []
        num_proposal_output = 0.0
        for proposal_bbox_inst in proposals_rpn_unsup_k:
            # thresholding
            if psedo_label_method == "thresholding":
                proposal_bbox_inst = self.threshold_bbox(
                    proposal_bbox_inst, thres=cur_threshold, proposal_type=proposal_type
                )
            else:
                raise ValueError("Unkown pseudo label boxes methods")
            num_proposal_output += len(proposal_bbox_inst)
            list_instances.append(proposal_bbox_inst)
        num_proposal_output = num_proposal_output / len(proposals_rpn_unsup_k)
        return list_instances, num_proposal_output

    def remove_label(self, label_data):
        for label_datum in label_data:
            if "instances" in label_datum.keys():
                del label_datum["instances"]
        return label_data

    def add_label(self, unlabled_data, label):
        for unlabel_datum, lab_inst in zip(unlabled_data, label):
            unlabel_datum["instances"] = lab_inst
        return unlabled_data

    # =====================================================
    # =================== Training Flow ===================
    # =====================================================

    def run_step_full_semisup(self):
        self._trainer.iter = self.iter
        assert self.model.training, "[UBTeacherTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        data = next(self._trainer._data_loader_iter)
        # data_q and data_k from different augmentations (q:strong, k:weak)
        # label_strong, label_weak, unlabed_strong, unlabled_weak
        label_data_q, label_data_k, unlabel_data_q, unlabel_data_k = data
        data_time = time.perf_counter() - start

        # initialize or update teacher model
        if self.iter == min(self.cfg.SEMISUPNET.BURN_UP_STEP, self.cfg.CONSISTENCY.START):  # first update copy the the whole model
            self._update_teacher_model(keep_rate=0.00)

        elif ((self.iter - min(self.cfg.SEMISUPNET.BURN_UP_STEP, self.cfg.CONSISTENCY.START)) % self.cfg.SEMISUPNET.TEACHER_UPDATE_ITER == 0) \
                and self.iter > min(self.cfg.SEMISUPNET.BURN_UP_STEP, self.cfg.CONSISTENCY.START):
            self._update_teacher_model(keep_rate=self.cfg.SEMISUPNET.EMA_KEEP_RATE)

        # burn-in stage (supervised training with labeled data)
        if self.iter < self.cfg.SEMISUPNET.BURN_UP_STEP:

            # input both strong and weak supervised data into model
            label_data_q.extend(label_data_k)
            record_dict, _, _, _ = self.model(label_data_q, branch="supervised")

            # weight losses
            loss_dict = {}
            for key in record_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = record_dict[key] * 1
            losses = sum(loss_dict.values())

        else: # pseudo-label branch
            if self.iter == self.cfg.SEMISUPNET.BURN_UP_STEP:
                # update copy the the whole model
                self._update_teacher_model(keep_rate=0.00)

            elif (
                self.iter - self.cfg.SEMISUPNET.BURN_UP_STEP
            ) % self.cfg.SEMISUPNET.TEACHER_UPDATE_ITER == 0:
                self._update_teacher_model(keep_rate=self.cfg.SEMISUPNET.EMA_KEEP_RATE)

            record_dict = {}
            #  generate the pseudo-label using teacher model
            # note that we do not convert to eval mode, as 1) there is no gradient computed in
            # teacher model and 2) batch norm layers are not updated as well
            with torch.no_grad():
                (
                    _,
                    proposals_rpn_unsup_k,
                    proposals_roih_unsup_k,
                    _,
                ) = self.model_teacher(unlabel_data_k, branch="unsup_data_weak")

            #  Pseudo-labeling
            cur_threshold = self.cfg.SEMISUPNET.BBOX_THRESHOLD

            joint_proposal_dict = {}
            joint_proposal_dict["proposals_rpn"] = proposals_rpn_unsup_k
            (
                pesudo_proposals_rpn_unsup_k,
                nun_pseudo_bbox_rpn,
            ) = self.process_pseudo_label(
                proposals_rpn_unsup_k, cur_threshold, "rpn", "thresholding"
            )
            joint_proposal_dict["proposals_pseudo_rpn"] = pesudo_proposals_rpn_unsup_k
            # Pseudo_labeling for ROI head (bbox location/objectness)
            pesudo_proposals_roih_unsup_k, _ = self.process_pseudo_label(
                proposals_roih_unsup_k, cur_threshold, "roih", "thresholding"
            )
            joint_proposal_dict["proposals_pseudo_roih"] = pesudo_proposals_roih_unsup_k

            #  add pseudo-label to unlabeled data
            unlabel_data_q = self.remove_label(unlabel_data_q)
            unlabel_data_k = self.remove_label(unlabel_data_k)

            unlabel_data_q = self.add_label(
                unlabel_data_q, joint_proposal_dict["proposals_pseudo_roih"]
            )
            unlabel_data_k = self.add_label(
                unlabel_data_k, joint_proposal_dict["proposals_pseudo_roih"]
            )

            all_label_data = label_data_q + label_data_k
            all_unlabel_data = unlabel_data_q

            # supervised branch
            with autocast():
                if self.cfg.SEMISUPNET.SUPERVISED_CROSS_LEVEL: # cross-level or not
                    record_all_label_data, _, _, _ = self.model(all_label_data, branch="supervised_cross_level")

                else:
                    # input number of samples: 2x batch_size (contains label_data_q + label_data_k)
                    record_all_label_data, _, _, _ = self.model(all_label_data, branch='supervised')
                record_dict.update(record_all_label_data)

                # pseudo-label branch
                if self.cfg.SEMISUPNET.PSEUDO_LABEL_CROSS_LEVEL: # cross-level or not
                    record_all_unlabel_data, _, _, _ = self.model(all_unlabel_data, branch="supervised_cross_level")
                else:
                    # input number of samples: 2x batch_size (contains label_data_q + label_data_k)
                    record_all_unlabel_data, _, _, _ = self.model(all_unlabel_data, branch='supervised')

            new_record_all_unlabel_data = {}
            for key in record_all_unlabel_data.keys():
                new_record_all_unlabel_data[key + "_pseudo"] = record_all_unlabel_data[
                    key
                ]
            record_dict.update(new_record_all_unlabel_data)
        with autocast():
            if self.cfg.CONSISTENCY.ROI_LEVEL and self.iter >= self.cfg.CONSISTENCY.START and self.iter <= self.cfg.CONSISTENCY.END:
                # get proposals from teacher model
                proposals = self.model_teacher(unlabel_data_k, branch='get_proposals')

                # get soft target
                teacher_predictions_cls, teacher_predictions_reg = self.model_teacher(unlabel_data_k,
                                                                                                        branch='roi_consistency_all_levels_heuristic',
                                                                                                        proposals=proposals)
                # Train student with all levels of features
                student_predictions_cls, student_predictions_reg = self.model(unlabel_data_q, branch='roi_consistency_all_levels',
                                                                              proposals=proposals)

                cls_loss, reg_loss, num_proposals, num_proposals_reg = self.postprocessing(student_predictions_cls,
                                                                                           student_predictions_reg,
                                                                                           teacher_predictions_cls,
                                                                                           teacher_predictions_reg)

                record_dict['num_proposals'] = num_proposals
                consistency_dict = dict(loss_cls_consistency=cls_loss, loss_reg_consistency=reg_loss)
                record_dict.update(consistency_dict)

                if self.cfg.WEIGHT_SCHEDULE.REWEIGHT:
                    w = self._weight_schedule()
                    record_dict['loss_cls_consistency'] = w * record_dict['loss_cls_consistency']
                    record_dict['loss_cls_pseudo'] = (1 - w) * record_dict['loss_cls_pseudo']
                    record_dict['loss_rpn_cls_pseudo'] = (1 - w) * record_dict['loss_rpn_cls_pseudo']
                    record_dict['consistency_reweight'] = w

            # weight losses
            loss_dict = {}
            for key in record_dict.keys():
                if key[:4] == "loss":
                    if key == "loss_rpn_loc_pseudo" or key == "loss_box_reg_pseudo":
                        # pseudo bbox regression <- 0
                        loss_dict[key] = record_dict[key] * 0
                    elif key[-6:] == "pseudo":  # unsupervised loss
                        loss_dict[key] = (
                            record_dict[key] * self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT
                        )
                    else:  # supervised loss
                        loss_dict[key] = record_dict[key] * 1

                losses = sum(loss_dict.values())

            metrics_dict = record_dict
            metrics_dict["data_time"] = data_time
            self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        # with AMP
        self._trainer.grad_scaler.scale(losses).backward()
        self._trainer.grad_scaler.step(self.optimizer)
        self._trainer.grad_scaler.update()
        #losses.backward()
        #self.optimizer.step()


    def compute_cls_loss(self, student_scores, teacher_scores):
        if self.cfg.CONSISTENCY.CLS == 'MSE':
            criterion = torch.nn.MSELoss(reduction='none')
        elif self.cfg.CONSISTENCY.CLS == 'KL':
            criterion = torch.nn.KLDivLoss(reduction='none')
        else:
            pass
        if self.cfg.CONSISTENCY.CLS == 'KL':
            loss = criterion(student_scores.log(), teacher_scores.detach())
        else:
            loss = criterion(student_scores, teacher_scores.detach())
        return loss

    def compute_reg_loss(self, student_offsets, teacher_offsets):
        if self.cfg.CONSISTENCY.REG == 'MSE':
            criterion = torch.nn.MSELoss(reduction='none')
        elif self.cfg.CONSISTENCY.REG == 'SmoothL1':
            criterion = torch.nn.SmoothL1Loss(reduction='none')

        loss = criterion(student_offsets, teacher_offsets.detach())

        return loss

    def _weight_schedule(self):
        t1 = self.cfg.WEIGHT_SCHEDULE.T1
        t2 = self.cfg.WEIGHT_SCHEDULE.T2
        if self.iter <= t1:
            w = self._cos_reweight(0, t1, self.cfg.WEIGHT_SCHEDULE.W1, 1)
        elif self.iter > t1 and self.iter < t2:
            w = self.cfg.WEIGHT_SCHEDULE.W1
        elif self.iter >= t2:
            w = self._cos_reweight(t2, self.cfg.CONSISTENCY.END, self.cfg.WEIGHT_SCHEDULE.W2, self.cfg.WEIGHT_SCHEDULE.W1)
        return w

    def _cos_reweight(self, start_iter, end_iter, eta_min, eta_max):
        pi = 3.14
        f_k = eta_min + 0.5 * (eta_max - eta_min) * (math.cos(((self.iter - start_iter) / (end_iter - start_iter)) * pi) + 1)
        if self.iter <= end_iter:
            weight = f_k
        else:
            weight = 1.0
        return weight

    def _cos_coefficient(self, start_iter, end_iter, total_proposals):
        pi = 3.14
        f_k = 0.5 * (math.cos(((self.iter - start_iter) / end_iter) * pi) + 1)
        if self.iter <= self.cfg.CONSISTENCY.RAMP_END:
            ramped_proposals = int(f_k * (total_proposals - self.cfg.CONSISTENCY.MIN_PROPOSAL)) + self.cfg.CONSISTENCY.MIN_PROPOSAL
        else:
            ramped_proposals = self.cfg.CONSISTENCY.MIN_PROPOSAL
        ohem_proposals = int(self.cfg.CONSISTENCY.MIN_PROPOSAL)
        num_proposals = max(ramped_proposals, ohem_proposals)
        return num_proposals


    def postprocessing(self, student_predictions_cls, student_predictions_reg, teacher_predictions_cls,
                       teacher_predictions_reg):
        student_scores = [F.softmax(s, dim=-1) + 1e-7 for s in student_predictions_cls]
        teacher_predictions_cls = F.softmax(teacher_predictions_cls, dim=-1) + 1e-7

        student_offsets = student_predictions_reg

        if self.cfg.CONSISTENCY.PAD_TYPE == 'min_proposal':
            num_proposals = self.cfg.CONSISTENCY.MIN_PROPOSAL
        elif self.cfg.CONSISTENCY.PAD_TYPE == 'ramp-to-proposal':
            num_proposals = self._cos_coefficient(self.cfg.CONSISTENCY.START, self.cfg.CONSISTENCY.RAMP_END,
                                                  teacher_predictions_cls.shape[0])
        else:
            num_proposals = teacher_predictions_cls.shape[0]

        cls_loss_all_levels = []
        reg_loss_all_levels = []
        num_proposals_list = []
        num_proposals_reg_list = []

        for c, r in zip(student_scores, student_offsets):
            if self.cfg.CONSISTENCY.APPLY_CLS:
                unreduced_cls_loss_per_level = self.compute_cls_loss(c, teacher_predictions_cls)

                # hard example mining
                cls_loss_per_level = torch.sum(unreduced_cls_loss_per_level, dim=-1)

                sorted_losses, indices = torch.sort(cls_loss_per_level, descending=True)
                actual_num_proposals = num_proposals
                cls_loss = sorted_losses[:num_proposals]
                num_proposals_list.append(actual_num_proposals)


                cls_loss = torch.mean(cls_loss)
                cls_loss_all_levels.append(cls_loss)

            if self.cfg.CONSISTENCY.APPLY_REG:
                if self.cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG:  # class agnostic regressor
                    max_scores, max_class = torch.max(teacher_predictions_cls, dim=-1)
                    fg_mask = (max_class != 80)
                    fg_proposal_rows = \
                    torch.arange(teacher_predictions_cls.shape[0], device=teacher_predictions_cls.device)[fg_mask]

                    fg_r = r[fg_proposal_rows, :]
                    fg_teacher = teacher_predictions_reg[fg_proposal_rows, :]
                else:  # class-specific regressor
                    max_scores, max_class = torch.max(teacher_predictions_cls, dim=-1)
                    fg_mask = (max_class != 80)
                    fg_classes = max_class[fg_mask]
                    fg_class_cols = 4 * fg_classes[:, None] + torch.arange(4, device=fg_classes.device)
                    fg_proposal_rows = \
                    torch.arange(teacher_predictions_cls.shape[0], device=teacher_predictions_cls.device)[fg_mask]

                    fg_r = r[fg_proposal_rows[:, None], fg_class_cols]
                    fg_teacher = teacher_predictions_reg[fg_proposal_rows[:, None], fg_class_cols]

                if fg_r.shape[0] == 0:
                    continue

                reg_loss_per_level = self.compute_reg_loss(fg_r, fg_teacher)
                reg_loss = torch.mean(reg_loss_per_level)
                reg_loss_all_levels.append(reg_loss)
                num_proposals_reg_list.append(fg_teacher.shape[0])

        if len(cls_loss_all_levels) != 0:
            cls_loss = torch.mean(torch.stack(cls_loss_all_levels))
        else:
            cls_loss = 0.0

        if len(reg_loss_all_levels) != 0:
            reg_loss = torch.mean(torch.stack(reg_loss_all_levels))
        else:
            reg_loss = 0.0

        num_proposals = sum(num_proposals_list) / len(num_proposals_list)
        if len(num_proposals_reg_list) == 0:
            num_proposals_reg = 0
        else:
            num_proposals_reg = sum(num_proposals_reg_list) / len(num_proposals_reg_list)

        return cls_loss, reg_loss, num_proposals, num_proposals_reg

    def _write_metrics(self, metrics_dict: dict):
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }

        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)
        # all_hg_dict = comm.gather(hg_dict)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                # data_time among workers can have high variance. The actual latency
                # caused by data_time is the maximum among workers.
                data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }

            # append the list
            loss_dict = {}
            for key in metrics_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = metrics_dict[key]

            total_losses_reduced = sum(loss for loss in loss_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)

    @torch.no_grad()
    def _update_teacher_model(self, keep_rate=0.996):
        if comm.get_world_size() > 1:
            student_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
        else:
            student_model_dict = self.model.state_dict()

        new_teacher_dict = OrderedDict()
        for key, value in self.model_teacher.state_dict().items():
            if key in student_model_dict.keys():
                new_teacher_dict[key] = (
                    student_model_dict[key] * (1 - keep_rate) + value * keep_rate
                )
            else:
                raise Exception("{} is not found in student model".format(key))

        self.model_teacher.load_state_dict(new_teacher_dict)

    @torch.no_grad()
    def _copy_main_model(self):
        # initialize all parameters
        if comm.get_world_size() > 1:
            rename_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
            self.model_teacher.load_state_dict(rename_model_dict)
        else:
            self.model_teacher.load_state_dict(self.model.state_dict())

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name)

    def build_hooks(self):
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )

        def test_and_save_results_student():
            self._last_eval_results_student = self.test(self.cfg, self.model)
            _last_eval_results_student = {
                k + "_student": self._last_eval_results_student[k]
                for k in self._last_eval_results_student.keys()
            }
            return _last_eval_results_student

        def test_and_save_results_teacher():
            self._last_eval_results_teacher = self.test(self.cfg, self.model_teacher)
            return self._last_eval_results_teacher

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results_student))
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results_teacher))

        if cfg.TEST.VAL_LOSS:  # default is True # save training time if not applied
            ret.append(
                LossEvalHook(
                    cfg.TEST.EVAL_PERIOD,
                    self.model,
                    build_detection_test_loader(
                        self.cfg,
                        self.cfg.DATASETS.TEST[0],
                        DatasetMapper(self.cfg, True),
                    ),
                    model_output="loss_proposal",
                    model_name="student",
                )
            )

            ret.append(
                LossEvalHook(
                    cfg.TEST.EVAL_PERIOD,
                    self.model_teacher,
                    build_detection_test_loader(
                        self.cfg,
                        self.cfg.DATASETS.TEST[0],
                        DatasetMapper(self.cfg, True),
                    ),
                    model_output="loss_proposal",
                    model_name="",
                )
            )

        if comm.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

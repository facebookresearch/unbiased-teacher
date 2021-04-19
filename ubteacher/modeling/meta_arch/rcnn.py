# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN


@META_ARCH_REGISTRY.register()
class TwoStagePseudoLabGeneralizedRCNN(GeneralizedRCNN):
    def forward(
        self, batched_inputs, branch="supervised", given_proposals=None, val_mode=False, **kwargs
    ):
        if (not self.training) and (not val_mode):
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if branch == "supervised":
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images, features, proposals_rpn, gt_instances, branch=branch
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None

        if branch == "supervised_cross_level":
            proposals_rpn, proposal_losses = self.proposal_generator(images, features, gt_instances)
            _, detector_losses = self.roi_heads(
                images, features, proposals_rpn, gt_instances, branch='cross_level')

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None

        elif branch == 'get_proposals':
            proposals_rpn, _ = self.proposal_generator(images, features, None, compute_loss=False)
            return proposals_rpn

        elif branch == 'roi_consistency_all_levels':
            # 1. generate proposals
            # proposals_rpn, _ = self.proposal_generator(images, features, None, compute_loss=False)
            proposals_rpn = kwargs['proposals']
            level_assignments_template = self.roi_heads.box_pooler.get_level_assignments(
                [x.proposal_boxes for x in proposals_rpn])
            level_assignments_template = torch.zeros_like(level_assignments_template)
            collected_predictions_cls = []
            collected_predictions_bbox = []
            for i in range(len(self.roi_heads.box_in_features)):
                level_assignments = level_assignments_template + i  # update level assignments
                current_predictions = self.roi_heads.predict_with_assignments(features, proposals_rpn,
                                                                              level_assignments, i)
                collected_predictions_cls.append(current_predictions[0])
                collected_predictions_bbox.append(current_predictions[1])
            return collected_predictions_cls, collected_predictions_bbox

        elif branch == 'roi_consistency_all_levels_heuristic':
            # 1. generate proposals
            # proposals_rpn, _ = self.proposal_generator(images, features, None, compute_loss=False)
            proposals_rpn = kwargs['proposals']
            predictions = self.roi_heads.forward_without_sampling(features, proposals_rpn)
            return predictions[0], predictions[1]


        elif branch == "unsup_data_weak":
            # Region proposal network
            proposals_rpn, _ = self.proposal_generator(
                images, features, None, compute_loss=False
            )

            # roi_head lower branch (keep this for further production)  # notice that we do not use any target in ROI head to do inference !
            proposals_roih, ROI_predictions = self.roi_heads(
                images,
                features,
                proposals_rpn,
                targets=None,
                compute_loss=False,
                branch=branch,
            )

            return {}, proposals_rpn, proposals_roih, ROI_predictions

        elif branch == "val_loss":

            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances, compute_val_loss=True
            )

            # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images,
                features,
                proposals_rpn,
                gt_instances,
                branch=branch,
                compute_val_loss=True,
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None

    # def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
    #     assert not self.training

    #     images = self.preprocess_image(batched_inputs)
    #     features = self.backbone(images.tensor)

    #     if detected_instances is None:
    #         if self.proposal_generator:
    #             proposals, _ = self.proposal_generator(images, features, None)
    #         else:
    #             assert "proposals" in batched_inputs[0]
    #             proposals = [x["proposals"].to(self.device) for x in batched_inputs]

    #         results, _ = self.roi_heads(images, features, proposals, None)
    #     else:
    #         detected_instances = [x.to(self.device) for x in detected_instances]
    #         results = self.roi_heads.forward_with_given_boxes(
    #             features, detected_instances
    #         )

    #     if do_postprocess:
    #         return GeneralizedRCNN._postprocess(
    #             results, batched_inputs, images.image_sizes
    #         )
    #     else:
    #         return results

# Copyright (c) Tencent Inc. All rights reserved.
from typing import List, Tuple, Union
import copy
import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import gradcheck
from mmdet.structures import OptSampleList, SampleList
from mmdet.models.utils import (multi_apply, unpack_gt_instances,
                                filter_scores_and_topk)
from mmdet.utils import OptConfigType, InstanceList, OptInstanceList
from mmengine.dist import get_dist_info

from typing import List, Optional, Tuple, Union, Sequence
from mmengine.structures import InstanceData
from mmengine.config import ConfigDict
from mmyolo.models.detectors import YOLODetector
from mmyolo.registry import MODELS
from mmyolo.models.utils import gt_instances_preprocess
import os

import torch.nn.functional as F
import joblib
import json

from torchvision.ops.boxes import box_iou

@MODELS.register_module()
class YOLOWorldDetector(YOLODetector):
    """Implementation of YOLOW Series"""

    def __init__(self,
                 *args,
                 mm_neck: bool = False,
                 num_train_classes=80,
                 num_test_classes=80,
                 **kwargs) -> None:
        self.mm_neck = mm_neck
        self.num_train_classes = num_train_classes
        self.num_test_classes = num_test_classes
        super().__init__(*args, **kwargs)

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples."""
        self.bbox_head.num_classes = self.num_train_classes
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        losses = self.bbox_head.loss(img_feats, txt_feats, batch_data_samples)
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.
        """

        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)

        # self.bbox_head.num_classes = self.num_test_classes
        self.bbox_head.num_classes = txt_feats[0].shape[0]
        results_list = self.bbox_head.predict(img_feats,
                                              txt_feats,
                                              batch_data_samples,
                                              rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def reparameterize(self, texts: List[List[str]]) -> None:
        # encode text embeddings into the detector
        self.texts = texts
        self.text_feats = self.backbone.forward_text(texts)

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.
        """
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        results = self.bbox_head.forward(img_feats, txt_feats)
        return results

    def extract_feat(
            self, batch_inputs: Tensor,
            batch_data_samples: SampleList) -> Tuple[Tuple[Tensor], Tensor]:
        """Extract features."""
        txt_feats = None
        if batch_data_samples is None:
            texts = self.texts
            txt_feats = self.text_feats
        elif isinstance(batch_data_samples,
                        dict) and 'texts' in batch_data_samples:
            texts = batch_data_samples['texts']
        elif isinstance(batch_data_samples, list) and hasattr(
                batch_data_samples[0], 'texts'):
            texts = [data_sample.texts for data_sample in batch_data_samples]
        elif hasattr(self, 'text_feats'):
            texts = self.texts
            txt_feats = self.text_feats
        else:
            raise TypeError('batch_data_samples should be dict or list.')
        if txt_feats is not None:
            # forward image only
            img_feats = self.backbone.forward_image(batch_inputs)
        else:
            img_feats, txt_feats = self.backbone(batch_inputs, texts)
        if self.with_neck:
            if self.mm_neck:
                img_feats = self.neck(img_feats, txt_feats)
            else:
                img_feats = self.neck(img_feats)
        return img_feats, txt_feats


@MODELS.register_module()
class SimpleYOLOWorldDetector(YOLODetector):
    """Implementation of YOLO World Series"""

    def __init__(self,
                 *args,
                 mm_neck: bool = False,
                 num_train_classes=80,
                 num_test_classes=80,
                 prompt_dim=512,
                 num_prompts=80,
                 embedding_path='',
                 reparameterized=False,
                 freeze_prompt=False,
                 use_mlp_adapter=False,
                 **kwargs) -> None:
        self.mm_neck = mm_neck
        self.num_training_classes = num_train_classes
        self.num_test_classes = num_test_classes
        self.prompt_dim = prompt_dim
        self.num_prompts = num_prompts
        self.reparameterized = reparameterized
        self.freeze_prompt = freeze_prompt
        self.use_mlp_adapter = use_mlp_adapter
        super().__init__(*args, **kwargs)

        if not self.reparameterized:
            if len(embedding_path) > 0:
                import numpy as np
                self.embeddings = torch.nn.Parameter(
                    torch.from_numpy(np.load(embedding_path)).float())
            else:
                # random init
                embeddings = nn.functional.normalize(torch.randn(
                    (num_prompts, prompt_dim)),
                                                     dim=-1)
                self.embeddings = nn.Parameter(embeddings)

            if self.freeze_prompt:
                self.embeddings.requires_grad = False
            else:
                self.embeddings.requires_grad = True

            if use_mlp_adapter:
                self.adapter = nn.Sequential(
                    nn.Linear(prompt_dim, prompt_dim * 2), nn.ReLU(True),
                    nn.Linear(prompt_dim * 2, prompt_dim))
            else:
                self.adapter = None

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples."""
        self.bbox_head.num_classes = self.num_training_classes
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        if self.reparameterized:
            losses = self.bbox_head.loss(img_feats, batch_data_samples)
        else:
            losses = self.bbox_head.loss(img_feats, txt_feats,
                                         batch_data_samples)
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.
            batch_inputs (Tensor): Input data (bs, 3, w, h).
            results_list (list): {
                bboxes (tensor): Predicted bboxes (num_bboxes, 4).
                labels (tensor): Predicted labels (num_bboxes, ).
                scores (tensor): Predicted scores (num_bboxes, ).
            }
        """

        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)

        self.bbox_head.num_classes = self.num_test_classes
        if self.reparameterized:
            results_list = self.bbox_head.predict(img_feats,
                                                  batch_data_samples,
                                                  rescale=rescale)
        else:
            results_list = self.bbox_head.predict(img_feats,
                                                  txt_feats,
                                                  batch_data_samples,
                                                  rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.
        """
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        if self.reparameterized:
            results = self.bbox_head.forward(img_feats)
        else:
            results = self.bbox_head.forward(img_feats, txt_feats)
        return results

    def extract_feat(
            self, batch_inputs: Tensor,
            batch_data_samples: SampleList) -> Tuple[Tuple[Tensor], Tensor]:
        """Extract features."""
        # only image features
        img_feats, _ = self.backbone(batch_inputs, None)

        if not self.reparameterized:
            # use embeddings
            txt_feats = self.embeddings[None]
            if self.adapter is not None:
                txt_feats = self.adapter(txt_feats) + txt_feats
                txt_feats = nn.functional.normalize(txt_feats, dim=-1, p=2)
            txt_feats = txt_feats.repeat(img_feats[0].shape[0], 1, 1)
        else:
            txt_feats = None
        if self.with_neck:
            if self.mm_neck:
                img_feats = self.neck(img_feats, txt_feats)
            else:
                img_feats = self.neck(img_feats)
        return img_feats, txt_feats


@MODELS.register_module()
class DecisionTreeYOLOWorld(YOLODetector):
    def __init__(self,
                 *args,
                 mm_neck: bool = False,
                 num_train_classes=80,
                 num_test_classes=80,
                 prompt_dim=512,
                 num_prompts=80,
                 embedding_path='',
                 reparameterized=False,
                 freeze_prompt=False,
                 use_mlp_adapter=False,
                 mode=None,
                 save_path=None,
                 tree_path=None,
                 att_path=None,
                 node_embedding_path=None,
                 connections=None,
                 **kwargs) -> None:
        self.mm_neck = mm_neck
        self.num_training_classes = num_train_classes
        self.num_test_classes = num_test_classes
        self.prompt_dim = prompt_dim
        self.num_prompts = num_prompts
        self.reparameterized = reparameterized
        self.freeze_prompt = freeze_prompt
        self.use_mlp_adapter = use_mlp_adapter
        self.mode=mode
        self.save_id = 1
        self.save_path = save_path
        self.connections = connections
        super().__init__(*args, **kwargs)
        
        if mode == 'trainable_tree':
            from .TrainableTree import TrainableTree
            self.decision_tree = TrainableTree(connections, num_train_classes, embedding_path)
        
        if mode == 'collection' and node_embedding_path is not None:
            import numpy as np
            self.node_embedding = torch.nn.Parameter(torch.from_numpy(np.load(node_embedding_path)).float())
        
        if not self.reparameterized:
            if len(embedding_path) > 0:
                import numpy as np
                self.embeddings = torch.nn.Parameter(
                    torch.from_numpy(np.load(embedding_path)).float())
            else:
                # random init
                embeddings = nn.functional.normalize(torch.randn(
                    (num_prompts, prompt_dim)),
                                                     dim=-1)
                self.embeddings = nn.Parameter(embeddings)

            if self.freeze_prompt:
                self.embeddings.requires_grad = False
            else:
                self.embeddings.requires_grad = True

            if use_mlp_adapter:
                self.adapter = nn.Sequential(
                    nn.Linear(prompt_dim, prompt_dim * 2), nn.ReLU(True),
                    nn.Linear(prompt_dim * 2, prompt_dim))
            else:
                self.adapter = None

    def save_msg(self, msg, path=None):
        def remove_negative(msg):
            # feature, head_arg, assigned_scores, flatten_cls_preds
            node_embedding = F.normalize(self.node_embedding, dim=-1, p=2)
            sim_scores = []
            features, head_args, labels, scores = msg['feature'], msg['head_arg'], \
                                                                    msg['assigned_scores'], msg['flatten_cls_preds']
            # Get similarity score
            for feature, head_arg in zip(features, head_args):
                sim_score = torch.einsum('bwc,kc->bwk', feature.float(), node_embedding.float())
                sim_score = (sim_score * head_arg[0] + head_arg[1]).sigmoid()
                sim_scores.append(sim_score)
            num_feature = node_embedding.shape[0]
            sim_scores = torch.cat(sim_scores, dim=1)
            fg_mask = (labels.sum(dim=-1) > 0)
            fg_scores, (ious, fg_labels) = sim_scores[fg_mask], torch.max(labels[fg_mask], dim=-1)
            x, y = fg_scores, fg_labels
            msg = {'x': x.detach().cpu(), 'y': y.detach().cpu(), 'ious': ious.detach().cpu()}
            return msg
        
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        if self.node_embedding is not None:
            msg = remove_negative(msg)
            
        if path is None:
            path = os.path.join(self.save_path, f'{self.save_id}.pth')
            self.save_id += 1
        torch.save(msg, path)
        # vaild file 
        try:
            torch.load(path)
        except:
            print(f'save msg error {path}')

    def collect_image_embeddings(self, batch_inputs: Tensor,
                                 batch_data_samples: SampleList):
        self.bbox_head.num_classes = self.num_training_classes
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        # box head 
        # loss func 
        outs = ([[], [], []] if self.training else [[], []])
        txt_feats = [txt_feats for _ in range(self.bbox_head.num_levels)]
        
        save_msg = {'feature': [], 'head_arg': []}
        for img_feat, txt_feat, cls_pred, reg_pred, cls_contrast\
            in zip(img_feats, txt_feats, self.bbox_head.head_module.cls_preds,
                   self.bbox_head.head_module.reg_preds, self.bbox_head.head_module.cls_contrasts):
            b, _, h, w = img_feat.shape
            cls_embed = cls_pred(img_feat)
            # cls_logit = cls_contrast(cls_embed, txt_feat)
            cls_contrast_x, cls_contrast_w = cls_embed, txt_feat
            cls_contrast_x = cls_contrast.norm(cls_contrast_x)
            cls_contrast_w = F.normalize(cls_contrast_w, dim=-1, p=2)
            cls_logit = torch.einsum('bchw,bkc->bkhw', cls_contrast_x, cls_contrast_w)
            cls_logit = cls_logit * cls_contrast.logit_scale.exp() + cls_contrast.bias
            # save msg
            save_msg['feature'].append(cls_contrast_x.permute(0, 2, 3, 1).reshape(b, -1, self.prompt_dim))
            save_msg['head_arg'].append((cls_contrast.logit_scale.exp(), cls_contrast.bias))
            
            bbox_dist_preds = reg_pred(img_feat)
            if self.bbox_head.head_module.reg_max > 1:
                bbox_dist_preds = bbox_dist_preds.reshape(
                    [-1, 4, self.bbox_head.head_module.reg_max, h * w]).permute(0, 3, 1, 2)

                # TODO: The get_flops script cannot handle the situation of
                #  matmul, and needs to be fixed later
                # bbox_preds = bbox_dist_preds.softmax(3).matmul(self.proj)
                bbox_preds = bbox_dist_preds.softmax(3).matmul(
                    self.bbox_head.head_module.proj.view([-1, 1])).squeeze(-1)
                bbox_preds = bbox_preds.transpose(1, 2).reshape(b, -1, h, w)
            else:
                bbox_preds = bbox_dist_preds
            if self.training:
                # outs.append([cls_logit, bbox_preds, bbox_dist_preds])
                outs[0].append(cls_logit)
                outs[1].append(bbox_preds)
                outs[2].append(bbox_dist_preds)
            else:
                # outs.append([cls_logit, bbox_preds])
                outs[0].append(cls_logit)
                outs[1].append(bbox_preds)
        
        
        loss_inputs = tuple(outs) + (batch_data_samples['bboxes_labels'],
                              batch_data_samples['img_metas'])
        # loss by feat
        cls_scores, bbox_preds, bbox_dist_preds,\
        batch_gt_instances, batch_img_metas = loss_inputs
        
        # loss by feat func body
        num_imgs = len(batch_img_metas)

        current_featmap_sizes = [
            cls_score.shape[2:] for cls_score in cls_scores
        ]
        # If the shape does not equal, generate new one
        if current_featmap_sizes != self.bbox_head.featmap_sizes_train:
            self.bbox_head.featmap_sizes_train = current_featmap_sizes

            mlvl_priors_with_stride = self.bbox_head.prior_generator.grid_priors(
                self.bbox_head.featmap_sizes_train,
                dtype=cls_scores[0].dtype,
                device=cls_scores[0].device,
                with_stride=True)

            self.bbox_head.num_level_priors = [len(n) for n in mlvl_priors_with_stride]
            self.bbox_head.flatten_priors_train = torch.cat(mlvl_priors_with_stride,
                                                  dim=0)
            self.bbox_head.stride_tensor = self.bbox_head.flatten_priors_train[..., [2]]

        # gt info
        gt_info = gt_instances_preprocess(batch_gt_instances, num_imgs)
        gt_labels = gt_info[:, :, :1]
        gt_bboxes = gt_info[:, :, 1:]  # xyxy
        pad_bbox_flag = (gt_bboxes.sum(-1, keepdim=True) > 0).float()

        # pred info
        flatten_cls_preds = [
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                 self.bbox_head.num_classes)
            for cls_pred in cls_scores
        ]
        flatten_pred_bboxes = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        
        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_pred_bboxes = torch.cat(flatten_pred_bboxes, dim=1)
        flatten_pred_bboxes = self.bbox_head.bbox_coder.decode(
            self.bbox_head.flatten_priors_train[..., :2], flatten_pred_bboxes,
            self.bbox_head.stride_tensor[..., 0])

        assigned_result = self.bbox_head.assigner(
            (flatten_pred_bboxes.detach()).type(gt_bboxes.dtype),
            flatten_cls_preds.detach().sigmoid(), self.bbox_head.flatten_priors_train,
            gt_labels, gt_bboxes, pad_bbox_flag)
        assigned_scores = assigned_result['assigned_scores']
        save_msg['assigned_scores'] = assigned_scores
        save_msg['flatten_cls_preds'] = flatten_cls_preds
        
        self.save_msg(save_msg)

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples."""
        self.bbox_head.num_classes = self.num_training_classes
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                    batch_data_samples)
        if self.mode == 'loss':
            if self.reparameterized:
                losses = self.bbox_head.loss(img_feats, batch_data_samples)
            else:
                losses = self.bbox_head.loss(img_feats, txt_feats,
                                            batch_data_samples)
            return losses
        
        elif self.mode == 'collection':
            with torch.no_grad():
                self.collect_image_embeddings(batch_inputs, batch_data_samples)
        
            # original implementation
            if self.reparameterized:
                losses = self.bbox_head.loss(img_feats, batch_data_samples)
            else:
                losses = self.bbox_head.loss(img_feats, txt_feats,
                                            batch_data_samples)
            return losses
        
        elif self.mode == 'finetune':
            # head module
            self.bbox_head.num_classes = self.num_training_classes
            txt_feats = [txt_feats for _ in range(self.bbox_head.num_levels)]
            outs = multi_apply(self.predict_forward_single, img_feats, txt_feats,
                           self.bbox_head.head_module.cls_preds, 
                           self.bbox_head.head_module.reg_preds, 
                           self.bbox_head.head_module.cls_contrasts)
            loss_input = outs + (batch_data_samples['bboxes_labels'],
                              batch_data_samples['img_metas'])
            return self.loss_by_feat(*loss_input)
          
        elif self.mode == 'trainable_tree':
            self.bbox_head.num_classes = self.num_training_classes
            txt_feats = [txt_feats for _ in range(self.bbox_head.num_levels)]
            outs = multi_apply(self.prediction_forward_single_trainable_tree, img_feats, txt_feats,
                            self.bbox_head.head_module.cls_preds,
                            self.bbox_head.head_module.reg_preds,
                            self.bbox_head.head_module.cls_contrasts)
            loss_inputs = outs + (batch_data_samples['bboxes_labels'],
                              batch_data_samples['img_metas'])
            return self.bbox_head.loss_by_feat(*loss_inputs)
    
        elif self.mode == 'image_condition':
            # collection all image embedding which iou > 0.8
            with torch.no_grad():
                self.image_condition(batch_inputs, batch_data_samples)
            # original implementation
            if self.reparameterized:
                losses = self.bbox_head.loss(img_feats, batch_data_samples)
            else:
                losses = self.bbox_head.loss(img_feats, txt_feats,
                                            batch_data_samples)
            return losses
    
    def image_condition(self, batch_inputs: Tensor,
                              batch_data_samples: SampleList):
        self.bbox_head.num_classes = self.num_training_classes
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        # box head 
        # loss func 
        outs = ([[], [], []] if self.training else [[], []])
        txt_feats = [txt_feats for _ in range(self.bbox_head.num_levels)]
        
        save_msg = {'feature': [], 'head_arg': []}
        for img_feat, txt_feat, cls_pred, reg_pred, cls_contrast\
            in zip(img_feats, txt_feats, self.bbox_head.head_module.cls_preds,
                   self.bbox_head.head_module.reg_preds, self.bbox_head.head_module.cls_contrasts):
            b, _, h, w = img_feat.shape
            cls_embed = cls_pred(img_feat)
            # cls_logit = cls_contrast(cls_embed, txt_feat)
            cls_contrast_x, cls_contrast_w = cls_embed, txt_feat
            cls_contrast_x = cls_contrast.norm(cls_contrast_x)
            cls_contrast_w = F.normalize(cls_contrast_w, dim=-1, p=2)
            cls_logit = torch.einsum('bchw,bkc->bkhw', cls_contrast_x, cls_contrast_w)
            cls_logit = cls_logit * cls_contrast.logit_scale.exp() + cls_contrast.bias
            # save msg
            save_msg['feature'].append(cls_contrast_x.permute(0, 2, 3, 1).reshape(b, -1, self.prompt_dim))
            save_msg['head_arg'].append((cls_contrast.logit_scale.exp(), cls_contrast.bias))
            
            bbox_dist_preds = reg_pred(img_feat)
            if self.bbox_head.head_module.reg_max > 1:
                bbox_dist_preds = bbox_dist_preds.reshape(
                    [-1, 4, self.bbox_head.head_module.reg_max, h * w]).permute(0, 3, 1, 2)

                # TODO: The get_flops script cannot handle the situation of
                #  matmul, and needs to be fixed later
                # bbox_preds = bbox_dist_preds.softmax(3).matmul(self.proj)
                bbox_preds = bbox_dist_preds.softmax(3).matmul(
                    self.bbox_head.head_module.proj.view([-1, 1])).squeeze(-1)
                bbox_preds = bbox_preds.transpose(1, 2).reshape(b, -1, h, w)
            else:
                bbox_preds = bbox_dist_preds
            if self.training:
                # outs.append([cls_logit, bbox_preds, bbox_dist_preds])
                outs[0].append(cls_logit)
                outs[1].append(bbox_preds)
                outs[2].append(bbox_dist_preds)
            else:
                # outs.append([cls_logit, bbox_preds])
                outs[0].append(cls_logit)
                outs[1].append(bbox_preds)
        
        
        loss_inputs = tuple(outs) + (batch_data_samples['bboxes_labels'],
                              batch_data_samples['img_metas'])
        # loss by feat
        cls_scores, bbox_preds, bbox_dist_preds,\
        batch_gt_instances, batch_img_metas = loss_inputs
        
        # loss by feat func body
        num_imgs = len(batch_img_metas)

        current_featmap_sizes = [
            cls_score.shape[2:] for cls_score in cls_scores
        ]
        # If the shape does not equal, generate new one
        if current_featmap_sizes != self.bbox_head.featmap_sizes_train:
            self.bbox_head.featmap_sizes_train = current_featmap_sizes

            mlvl_priors_with_stride = self.bbox_head.prior_generator.grid_priors(
                self.bbox_head.featmap_sizes_train,
                dtype=cls_scores[0].dtype,
                device=cls_scores[0].device,
                with_stride=True)

            self.bbox_head.num_level_priors = [len(n) for n in mlvl_priors_with_stride]
            self.bbox_head.flatten_priors_train = torch.cat(mlvl_priors_with_stride,
                                                  dim=0)
            self.bbox_head.stride_tensor = self.bbox_head.flatten_priors_train[..., [2]]

        # gt info
        gt_info = gt_instances_preprocess(batch_gt_instances, num_imgs)
        gt_labels = gt_info[:, :, :1]
        gt_bboxes = gt_info[:, :, 1:]  # xyxy
        pad_bbox_flag = (gt_bboxes.sum(-1, keepdim=True) > 0).float()

        # pred info
        flatten_cls_preds = [
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                 self.bbox_head.num_classes)
            for cls_pred in cls_scores
        ]
        flatten_pred_bboxes = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        
        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_pred_bboxes = torch.cat(flatten_pred_bboxes, dim=1)
        flatten_pred_bboxes = self.bbox_head.bbox_coder.decode(
            self.bbox_head.flatten_priors_train[..., :2], flatten_pred_bboxes,
            self.bbox_head.stride_tensor[..., 0])
        features = torch.cat(save_msg['feature'], dim=1)
        
        self.save_image_embedding(self.get_match_result(features, gt_labels, gt_bboxes, 
                                                        flatten_pred_bboxes, pad_bbox_flag))
        
    def save_image_embedding(self, msg):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        path = os.path.join(self.save_path, f'{self.save_id}.pth')
        self.save_id += 1
        torch.save(msg, path)
        # vaild file 
        try:
            torch.load(path)
        except:
            print(f'save msg error {path}')
    
    def get_match_result(self, features, label_classes, label_boxes, pred_boxes, pad_bbox_flag):
        """_summary_

        Args:
            features (_type_): batch_size, num_features, prompt_dim
            label_classes (_type_): batch_size, num_gt, 1
            label_boxes (_type_):  batch_size, num_gt, 4
            pred_boxes (_type_):  batch_size, num_features, 4
            pad_bbox_flag (_type_):  batch_size, num_gt, 1
        """
        result = {}
        for feature, label_class, label_box, pred_box, pad_flag in zip(features, label_classes, 
                                                                       label_boxes, pred_boxes, 
                                                                       pad_bbox_flag):
            label_class, label_box = label_class[pad_flag.bool().squeeze(-1)], label_box[pad_flag.bool().squeeze(-1)]
            ious = box_iou(label_box, pred_box)
            positive_mask = (ious > 0.8)
            max_match = torch.argmax(ious, dim=-1)
            positive_mask.scatter_(1, max_match.unsqueeze(-1), True)
            for label_class, pos_mask in zip(label_class, positive_mask):
                if label_class.item() not in result:
                    result[int(label_class.item())] = []
                result[int(label_class.item())].append(feature[pos_mask])
        for key in result:
            result[key] = torch.cat(result[key], dim=0)
        
        return result
    
    def prediction_forward_single_trainable_tree(self, img_feat: Tensor, text_feats: Tensor,
                       cls_pred: nn.ModuleList, reg_pred: nn.ModuleList,
                       cls_contrast: nn.ModuleList) -> Tuple:
        """Forward feature of a single scale level."""
        b, _, h, w = img_feat.shape
        cls_embed = cls_contrast.norm(cls_pred(img_feat))
        # (B*H*W, C)
        embeding = cls_embed.permute(0, 2, 3, 1).reshape(-1, self.prompt_dim)
        cls_logit, decisions = self.decision_tree(embeding, [cls_contrast.logit_scale.exp(), cls_contrast.bias])
        cls_logit = cls_logit.reshape(b, h, w, self.num_training_classes).permute(0, 3, 1, 2)
        if not self.training:
            num_att = decisions.shape[-1]
            decisions = decisions.reshape(b, h, w, num_att)
        
        bbox_dist_preds = reg_pred(img_feat)
        if self.bbox_head.head_module.reg_max > 1:
            bbox_dist_preds = bbox_dist_preds.reshape(
                [-1, 4, self.bbox_head.head_module.reg_max, h * w]).permute(0, 3, 1, 2)

            # TODO: The get_flops script cannot handle the situation of
            #  matmul, and needs to be fixed later
            # bbox_preds = bbox_dist_preds.softmax(3).matmul(self.proj)
            bbox_preds = bbox_dist_preds.softmax(3).matmul(
                self.bbox_head.head_module.proj.view([-1, 1])).squeeze(-1)
            bbox_preds = bbox_preds.transpose(1, 2).reshape(b, -1, h, w)
        else:
            bbox_preds = bbox_dist_preds
        if self.training:
            return cls_logit, bbox_preds, bbox_dist_preds
        else:
            return cls_logit, bbox_preds, decisions
    # head loss_by_feat
    def loss_by_feat(self,
            cls_scores: Sequence[Tensor],
            bbox_preds: Sequence[Tensor],
            bbox_dist_preds: Sequence[Tensor],
            decision_tree: Sequence[Tensor],
            batch_gt_instances: Sequence[InstanceData],
            batch_img_metas: Sequence[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (Sequence[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_priors * num_classes.
            bbox_preds (Sequence[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_priors * 4.
            bbox_dist_preds (Sequence[Tensor]): Box distribution logits for
                each scale level with shape (bs, reg_max + 1, H*W, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
        Returns:
            dict[str, Tensor]: A dictionary of losses.
        """
        def update_assign_result(assigned_result, flatten_decision_tree):
            assigned_scores = assigned_result['assigned_scores']
            mask = torch.zeros_like(assigned_scores, dtype=torch.bool)
            mask = torch.cat([mask, torch.zeros_like(mask[..., :1])], dim=-1)
            mask.scatter_(-1, flatten_decision_tree.unsqueeze(-1), 1)
            mask = mask[..., :-1]
            # assign result 
            assigned_scores[~mask] = 0
            fg_mask_pre_prior = (assigned_scores.sum(-1) > 0)
            assigned_result['fg_mask_pre_prior'] = fg_mask_pre_prior
            assigned_result['assigned_scores'] = assigned_scores
            return assigned_result
            
        num_imgs = len(batch_img_metas)

        current_featmap_sizes = [
            cls_score.shape[2:] for cls_score in cls_scores
        ]
        # If the shape does not equal, generate new one
        if current_featmap_sizes != self.bbox_head.featmap_sizes_train:
            self.bbox_head.featmap_sizes_train = current_featmap_sizes

            mlvl_priors_with_stride = self.bbox_head.prior_generator.grid_priors(
                self.bbox_head.featmap_sizes_train,
                dtype=cls_scores[0].dtype,
                device=cls_scores[0].device,
                with_stride=True)

            self.bbox_head.num_level_priors = [len(n) for n in mlvl_priors_with_stride]
            self.bbox_head.flatten_priors_train = torch.cat(mlvl_priors_with_stride,
                                                  dim=0)
            self.bbox_head.stride_tensor = self.bbox_head.flatten_priors_train[..., [2]]

        # gt info
        gt_info = gt_instances_preprocess(batch_gt_instances, num_imgs)
        gt_labels = gt_info[:, :, :1]
        gt_bboxes = gt_info[:, :, 1:]  # xyxy
        pad_bbox_flag = (gt_bboxes.sum(-1, keepdim=True) > 0).float()

        # pred info
        flatten_cls_preds = [
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                 self.bbox_head.num_classes)
            for cls_pred in cls_scores
        ]
        flatten_pred_bboxes = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        # (bs, n, 4 * reg_max)
        flatten_pred_dists = [
            bbox_pred_org.reshape(num_imgs, -1, self.bbox_head.head_module.reg_max * 4)
            for bbox_pred_org in bbox_dist_preds
        ]
        # (bs, W*H)
        flatten_decision_tree = [
            decision_tree_one_level.reshape(num_imgs, -1)
            for decision_tree_one_level in decision_tree
        ]
        flatten_dist_preds = torch.cat(flatten_pred_dists, dim=1)
        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_pred_bboxes = torch.cat(flatten_pred_bboxes, dim=1)
        flatten_pred_bboxes = self.bbox_head.bbox_coder.decode(
            self.bbox_head.flatten_priors_train[..., :2], flatten_pred_bboxes,
            self.bbox_head.stride_tensor[..., 0])
        flatten_decision_tree = torch.cat(flatten_decision_tree, dim=1)

        assigned_result = self.bbox_head.assigner(
            (flatten_pred_bboxes.detach()).type(gt_bboxes.dtype),
            flatten_cls_preds.detach().sigmoid(), self.bbox_head.flatten_priors_train,
            gt_labels, gt_bboxes, pad_bbox_flag)

        assigned_result = update_assign_result(assigned_result, flatten_decision_tree)
        assigned_bboxes = assigned_result['assigned_bboxes']
        assigned_scores = assigned_result['assigned_scores']
        fg_mask_pre_prior = assigned_result['fg_mask_pre_prior']

        assigned_scores_sum = assigned_scores.sum().clamp(min=1)

        loss_cls = self.bbox_head.loss_cls(flatten_cls_preds, assigned_scores).sum()
        loss_cls /= assigned_scores_sum

        # rescale bbox
        assigned_bboxes /= self.bbox_head.stride_tensor
        flatten_pred_bboxes /= self.bbox_head.stride_tensor

        # select positive samples mask
        num_pos = fg_mask_pre_prior.sum()
        if num_pos > 0:
            # when num_pos > 0, assigned_scores_sum will >0, so the loss_bbox
            # will not report an error
            # iou loss
            prior_bbox_mask = fg_mask_pre_prior.unsqueeze(-1).repeat([1, 1, 4])
            pred_bboxes_pos = torch.masked_select(
                flatten_pred_bboxes, prior_bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = torch.masked_select(
                assigned_bboxes, prior_bbox_mask).reshape([-1, 4])
            bbox_weight = torch.masked_select(assigned_scores.sum(-1),
                                              fg_mask_pre_prior).unsqueeze(-1)
            loss_bbox = self.bbox_head.loss_bbox(
                pred_bboxes_pos, assigned_bboxes_pos,
                weight=bbox_weight) / assigned_scores_sum

            # dfl loss
            pred_dist_pos = flatten_dist_preds[fg_mask_pre_prior]
            assigned_ltrb = self.bbox_head.bbox_coder.encode(
                self.bbox_head.flatten_priors_train[..., :2] / self.bbox_head.stride_tensor,
                assigned_bboxes,
                max_dis=self.bbox_head.head_module.reg_max - 1,
                eps=0.01)
            assigned_ltrb_pos = torch.masked_select(
                assigned_ltrb, prior_bbox_mask).reshape([-1, 4])
            loss_dfl = self.bbox_head.loss_dfl(pred_dist_pos.reshape(
                -1, self.bbox_head.head_module.reg_max),
                                     assigned_ltrb_pos.reshape(-1),
                                     weight=bbox_weight.expand(-1,
                                                               4).reshape(-1),
                                     avg_factor=assigned_scores_sum)
        else:
            loss_bbox = flatten_pred_bboxes.sum() * 0
            loss_dfl = flatten_pred_bboxes.sum() * 0
        if self.bbox_head.world_size == -1:
            _, world_size = get_dist_info()
        else:
            world_size = self.bbox_head.world_size
        return dict(loss_cls=loss_cls * num_imgs * world_size,
                    loss_bbox=loss_bbox * num_imgs * world_size,
                    loss_dfl=loss_dfl * num_imgs * world_size)
        
    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.
        """

        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)

        self.bbox_head.num_classes = self.num_test_classes
        if self.reparameterized:
            results_list = self.bbox_head.predict(img_feats,
                                                  batch_data_samples,
                                                  rescale=rescale)
        else:
            # head predict 
            batch_img_metas = [
                data_samples.metainfo for data_samples in batch_data_samples
            ] 
            # head module
            txt_feats = [txt_feats for _ in range(self.bbox_head.num_levels)]
            if self.mode == 'trainable_tree':
                outs = multi_apply(self.prediction_forward_single_trainable_tree, img_feats, txt_feats,
                            self.bbox_head.head_module.cls_preds,
                            self.bbox_head.head_module.reg_preds,
                            self.bbox_head.head_module.cls_contrasts)
                results_list = self.predict_by_feat(*outs, 
                                                     batch_img_metas=batch_img_metas,
                                                     rescale=rescale)
            elif self.mode == 'finetune':    
                outs = multi_apply(self.predict_forward_single, img_feats, txt_feats,
                            self.bbox_head.head_module.cls_preds, 
                            self.bbox_head.head_module.reg_preds, 
                            self.bbox_head.head_module.cls_contrasts)
                outs = self.update_outs_base_on_tree(outs)
                results_list = self.bbox_head.predict_by_feat(*outs, 
                                                        batch_img_metas=batch_img_metas,
                                                        rescale=rescale)
            # collection
            else:
                if self.reparameterized:
                    results_list = self.bbox_head.predict(img_feats,
                                                        batch_data_samples,
                                                        rescale=rescale)
                else:
                    results_list = self.bbox_head.predict(img_feats,
                                                        txt_feats[0],
                                                        batch_data_samples,
                                                        rescale=rescale)
            
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        decisions: List[Tensor],
                        objectnesses: Optional[List[Tensor]] = None,
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = True,
                        with_nms: bool = True) -> List[InstanceData]:
        assert len(cls_scores) == len(bbox_preds)
        if objectnesses is None:
            with_objectnesses = False
        else:
            with_objectnesses = True
            assert len(cls_scores) == len(objectnesses)

        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)

        multi_label = cfg.multi_label
        multi_label &= self.bbox_head.num_classes > 1
        cfg.multi_label = multi_label

        num_imgs = len(batch_img_metas)
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]

        # If the shape does not change, use the previous mlvl_priors
        if featmap_sizes != self.bbox_head.featmap_sizes:
            self.bbox_head.mlvl_priors = self.bbox_head.prior_generator.grid_priors(
                featmap_sizes,
                dtype=cls_scores[0].dtype,
                device=cls_scores[0].device)
            self.bbox_head.featmap_sizes = featmap_sizes
        flatten_priors = torch.cat(self.bbox_head.mlvl_priors)

        mlvl_strides = [
            flatten_priors.new_full(
                (featmap_size.numel() * self.bbox_head.num_base_priors, ), stride) for
            featmap_size, stride in zip(featmap_sizes, self.bbox_head.featmap_strides)
        ]
        flatten_stride = torch.cat(mlvl_strides)

        # flatten cls_scores, bbox_preds and objectness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  self.bbox_head.num_classes)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_decisions = [
            decision.permute(0, 2, 3, 1).reshape(num_imgs, -1, decision.shape[-1])
            for decision in decisions
        ]
        flatten_decisions = torch.cat(flatten_decisions, dim=1)
        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_decoded_bboxes = self.bbox_head.bbox_coder.decode(
            flatten_priors[None], flatten_bbox_preds, flatten_stride)

        if with_objectnesses:
            flatten_objectness = [
                objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
                for objectness in objectnesses
            ]
            flatten_objectness = torch.cat(flatten_objectness, dim=1).sigmoid()
        else:
            flatten_objectness = [None for _ in range(num_imgs)]
        # 8400
        # print(flatten_cls_scores.shape)
        results_list = []
        for (bboxes, scores, objectness,
             img_meta, decisions) in zip(flatten_decoded_bboxes, flatten_cls_scores,
                              flatten_objectness, batch_img_metas,
                              flatten_decisions):
            ori_shape = img_meta['ori_shape']
            scale_factor = img_meta['scale_factor']
            if 'pad_param' in img_meta:
                pad_param = img_meta['pad_param']
            else:
                pad_param = None

            score_thr = cfg.get('score_thr', -1)
            # yolox_style does not require the following operations
            if objectness is not None and score_thr > 0 and not cfg.get(
                    'yolox_style', False):
                conf_inds = objectness > score_thr
                bboxes = bboxes[conf_inds, :]
                scores = scores[conf_inds, :]
                objectness = objectness[conf_inds]

            if objectness is not None:
                # conf = obj_conf * cls_conf
                scores *= objectness[:, None]

            if scores.shape[0] == 0:
                empty_results = InstanceData()
                empty_results.bboxes = bboxes
                empty_results.scores = scores[:, 0]
                empty_results.labels = scores[:, 0].int()
                results_list.append(empty_results)
                continue

            nms_pre = cfg.get('nms_pre', 100000)
            if cfg.multi_label is False:
                scores, labels = scores.max(1, keepdim=True)
                scores, _, keep_idxs, results = filter_scores_and_topk(
                    scores,
                    score_thr,
                    nms_pre,
                    results=dict(labels=labels[:, 0]))
                labels = results['labels']
            else:
                scores, labels, keep_idxs, _ = filter_scores_and_topk(
                    scores, score_thr, nms_pre)

            results = InstanceData(scores=scores,
                                   labels=labels,
                                   bboxes=bboxes[keep_idxs])
            results.decisions = decisions[keep_idxs]
            if rescale:
                if pad_param is not None:
                    results.bboxes -= results.bboxes.new_tensor([
                        pad_param[2], pad_param[0], pad_param[2], pad_param[0]
                    ])
                results.bboxes /= results.bboxes.new_tensor(
                    scale_factor).repeat((1, 2))

            if cfg.get('yolox_style', False):
                # do not need max_per_img
                cfg.max_per_img = len(results)

            results = self.bbox_head._bbox_post_process(results=results,
                                              cfg=cfg,
                                              rescale=False,
                                              with_nms=with_nms,
                                              img_meta=img_meta)
            results.bboxes[:, 0::2].clamp_(0, ori_shape[1])
            results.bboxes[:, 1::2].clamp_(0, ori_shape[0])

            results_list.append(results)
        return results_list

        

    def update_outs_base_on_tree(self, outs):
        # eval mode
        cls_logits, category_ids = outs[0], outs[-1] 
        updated_cls_logits = []
        for cls_logit, cat_id in zip(cls_logits, category_ids):
            mask = torch.zeros_like(cls_logit, dtype=torch.bool).permute(0, 2, 3, 1)
            mask = torch.cat([mask, torch.zeros_like(mask[..., :1])], dim=-1)
            mask.scatter_(-1, cat_id.unsqueeze(-1), True)
            mask = mask[..., :-1].permute(0, 3, 1, 2)
            if cls_logit.dtype == torch.float16:
                cls_logit[~mask] = -65504
            else:
                cls_logit[~mask] = -1e5
            updated_cls_logits.append(cls_logit)
        return (updated_cls_logits,) + outs[1:-1]

    def predict_forward_single(self, img_feat: Tensor, txt_feat: Tensor,
                       cls_pred: nn.ModuleList, reg_pred: nn.ModuleList,
                       cls_contrast: nn.ModuleList) -> Tuple:
        """Forward feature of a single scale level."""
        b, _, h, w = img_feat.shape
        cls_embed = cls_pred(img_feat)
        cls_logit = cls_contrast(cls_embed, txt_feat)
        # decision tree
        normed_cls_embed = cls_contrast.norm(cls_embed)        
        bs, c, h, w = normed_cls_embed.shape
        normed_cls_embed = normed_cls_embed.permute(0, 2, 3, 1).reshape(bs, -1, c)
        # bs, h*w, num_att
        sim_score = torch.einsum('bwc,kc->bwk', normed_cls_embed.float(), self.att_embeddings)
        sim_score = (sim_score * cls_contrast.logit_scale.exp() + cls_contrast.bias).sigmoid()
        _, _, num_att = sim_score.shape
        with torch.no_grad():
            category_id = self.decision_tree.predict(sim_score.cpu().reshape(-1, num_att)).reshape(bs, h, w)
        category_id = torch.from_numpy(category_id).to(sim_score.device)
        # set bg to -1e5 
        # TODO original implementation, grad problem
        # mask = torch.zeros_like(cls_logit, dtype=torch.bool).permute(0, 2, 3, 1)
        # mask = torch.cat([mask, torch.zeros_like(mask[..., :1])], dim=-1)
        # mask.scatter_(-1, category_id.unsqueeze(-1), True)
        # mask = mask[..., :-1].permute(0, 3, 1, 2)
        # if cls_logit.dtype == torch.float16:
        #     cls_logit[~mask] = -65504
        # else:
        #     cls_logit[~mask] = -1e5
        
        bbox_dist_preds = reg_pred(img_feat)
        if self.bbox_head.head_module.reg_max > 1:
            bbox_dist_preds = bbox_dist_preds.reshape(
                [-1, 4, self.bbox_head.head_module.reg_max, h * w]).permute(0, 3, 1, 2)

            # TODO: The get_flops script cannot handle the situation of
            #  matmul, and needs to be fixed later
            # bbox_preds = bbox_dist_preds.softmax(3).matmul(self.proj)
            bbox_preds = bbox_dist_preds.softmax(3).matmul(
                self.bbox_head.head_module.proj.view([-1, 1])).squeeze(-1)
            bbox_preds = bbox_preds.transpose(1, 2).reshape(b, -1, h, w)
        else:
            bbox_preds = bbox_dist_preds
        if self.training:
            return cls_logit, bbox_preds, bbox_dist_preds, category_id
        else:
            return cls_logit, bbox_preds, category_id

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.
        """
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        if self.reparameterized:
            results = self.bbox_head.forward(img_feats)
        else:
            results = self.bbox_head.forward(img_feats, txt_feats)
        return results

    def extract_feat(
            self, batch_inputs: Tensor,
            batch_data_samples: SampleList) -> Tuple[Tuple[Tensor], Tensor]:
        """Extract features."""
        # only image features
        img_feats, _ = self.backbone(batch_inputs, None)

        if not self.reparameterized:
            # use embeddings
            txt_feats = self.embeddings[None]
            if self.adapter is not None:
                txt_feats = self.adapter(txt_feats) + txt_feats
                txt_feats = nn.functional.normalize(txt_feats, dim=-1, p=2)
            txt_feats = txt_feats.repeat(img_feats[0].shape[0], 1, 1)
        else:
            txt_feats = None
        if self.with_neck:
            if self.mm_neck:
                img_feats = self.neck(img_feats, txt_feats)
            else:
                img_feats = self.neck(img_feats)
        return img_feats, txt_feats
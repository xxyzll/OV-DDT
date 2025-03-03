_base_ = ('../../../third_party/mmyolo/configs/yolov8/'
          'yolov8_l_syncbn_fast_8xb16-500e_coco.py')

custom_imports = dict(imports=['yolo_world', 'yolo_world.Hooks.PrintGrad'], allow_failed_imports=False)

dataset_name = 'coco'
# new_bing_precise, gpt4o
model_name = 'gpt4o'
mode = "trainable_tree"
class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 
               'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']
# hyper-parameters
num_classes = len(class_names)
num_training_classes = len(class_names)
max_epochs = 80
close_mosaic_epochs = 10
save_epoch_intervals = 5
text_channels = 512
neck_embed_channels = [128, 256, _base_.last_stage_out_channels // 2]
neck_num_heads = [4, 8, _base_.last_stage_out_channels // 2 // 32]
base_lr = 0.0002
weight_decay = 0.05
train_batch_size_per_gpu = 16
# load_from = f'/home/xx/YOLO-World/work_dirs/prompt_finetune_clip/{dataset_name}_yolo_world_l/epoch_80.pth'
load_from =  '/home/xx/YOLO-World/work_dirs/yolo_world_v2_l_vlpan_bn_2e-4_80e_8gpus_prompt_tuning_coco/epoch_80.pth'
persistent_workers = False

# model settings
model = dict(type='DecisionTreeYOLOWorld',
             mm_neck=True,
             num_train_classes=num_training_classes,
             num_test_classes=num_classes,
             embedding_path=f'/home/xx/YOLO-World/data/texts/{dataset_name}_embeddings.npy',
             prompt_dim=text_channels,
             num_prompts=num_classes,  
             freeze_prompt=False,
             mode=mode,
             save_path=f'decision_tree/yolo_world_l/{dataset_name}',
             data_preprocessor=dict(type='YOLOv5DetDataPreprocessor'),
             tree_path=f'data/texts/yolo_world_l_{dataset_name}_{model_name}_decision_tree.joblib',
             att_path=f'data/texts/{dataset_name}_{model_name}.npy',
             connections=f'data/texts/yolo_world_l_{dataset_name}_{model_name}_decision_tree.json',
             node_embedding_path=f'/home/xx/YOLO-World/data/texts/{dataset_name}_{model_name}.npy',
             backbone=dict(_delete_=True,
                           type='MultiModalYOLOBackbone',
                           text_model=None,
                           image_model={{_base_.model.backbone}},
                           frozen_stages=4,
                           with_text_model=False),
             neck=dict(type='YOLOWorldPAFPN',
                       freeze_all=True,
                       guide_channels=text_channels,
                       embed_channels=neck_embed_channels,
                       num_heads=neck_num_heads,
                       block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConv')),
             bbox_head=dict(type='YOLOWorldHead',
                            head_module=dict(
                                type='YOLOWorldHeadModule',
                                freeze_all=True,
                                use_bn_head=True,
                                embed_dims=text_channels,
                                num_classes=num_training_classes)),
             train_cfg=dict(assigner=dict(num_classes=num_training_classes)))

# dataset settings
# coco_train_dataset = dict(type='YOLOv5CocoDataset',
#                           data_root='data/coco',
#                           ann_file='annotations/instances_train2017.json',
#                           data_prefix=dict(img='train2017/'),
#                           filter_cfg=dict(filter_empty_gt=False, min_size=32),
#                           pipeline=_base_.train_pipeline)

# dataset settings
coco_train_dataset = dict(type='YOLOv5CocoDataset',
                          data_root='data/coco',
                          ann_file='annotations/instances_train2017.json',
                          data_prefix=dict(img='train2017/'),
                          filter_cfg=dict(filter_empty_gt=False, min_size=32),
                          pipeline=_base_.train_pipeline)

train_dataloader = dict(persistent_workers=persistent_workers,
                        batch_size=train_batch_size_per_gpu,
                        collate_fn=dict(type='yolow_collate'),
                        dataset=coco_train_dataset)

coco_val_dataset = dict(type='YOLOv5CocoDataset',
                        data_root='data/coco',
                        ann_file='annotations/instances_val2017.json',
                        data_prefix=dict(img='val2017/'),
                        filter_cfg=dict(filter_empty_gt=False, min_size=32),
                        pipeline=_base_.test_pipeline)

val_dataloader = dict(dataset=coco_val_dataset)
test_dataloader = val_dataloader
# training settings
default_hooks = dict(param_scheduler=dict(scheduler_type='linear',
                                          lr_factor=0.01,
                                          max_epochs=max_epochs),
                     checkpoint=dict(max_keep_ckpts=-1,
                                     save_best=None,
                                     interval=save_epoch_intervals))
custom_hooks = [
    dict(type='EMAHook',
         ema_type='ExpMomentumEMA',
         momentum=0.0001,
         update_buffers=True,
         strict_load=False,
         priority=49),
    dict(type='mmdet.PipelineSwitchHook',
         switch_epoch=max_epochs - close_mosaic_epochs,
         switch_pipeline=_base_.train_pipeline_stage2)
]
train_cfg = dict(max_epochs=max_epochs,
                 val_interval=5,
                 dynamic_intervals=[((max_epochs - close_mosaic_epochs),
                                     _base_.val_interval_stage2)])

optim_wrapper = dict(optimizer=dict(
    _delete_=True,
    type='AdamW',
    lr=base_lr,
    weight_decay=weight_decay,
    batch_size_per_gpu=train_batch_size_per_gpu),
                     paramwise_cfg=dict(custom_keys={
                                            'backbone.text_model':
                                            dict(lr_mult=0.01),
                                            'logit_scale':
                                            dict(weight_decay=0.0),
                                            'embeddings':
                                            dict(weight_decay=0.0)
                                        }),
                     constructor='YOLOWv5OptimizerConstructor')

# evaluation settings
val_evaluator = dict(_delete_=True,
                     type='mmdet.CocoMetric',
                     proposal_nums=(100, 1, 10),
                     ann_file='data/coco/annotations/instances_val2017.json',
                     metric='bbox')
vis_backends=[dict(type='TensorboardVisBackend')]
visualizer = dict(
    name='visualizer',
    type='mmdet.DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend')
    ])

find_unused_parameters = True
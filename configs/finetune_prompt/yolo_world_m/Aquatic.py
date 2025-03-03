_base_ = ('../../../third_party/mmyolo/configs/yolov8/'
          'yolov8_m_syncbn_fast_8xb16-500e_coco.py')
custom_imports = dict(imports=['yolo_world'], allow_failed_imports=False)

dataset_name = 'Aquatic'
class_names = ['shark', 'fish', 'puffin', 'starfish', 'penguin', 'jellyfish', 'stingray']
# hyper-parameters
num_classes = len(class_names)
num_training_classes = len(class_names)
max_epochs = 80
close_mosaic_epochs = 10
save_epoch_intervals = 5
text_channels = 512
neck_embed_channels = [128, 256, _base_.last_stage_out_channels // 2]
neck_num_heads = [4, 8, _base_.last_stage_out_channels // 2 // 32]
base_lr = 0.096
weight_decay = 0.05
train_batch_size_per_gpu = 16
load_from = 'pretrained_models/yolo_world_v2_m_obj365v1_goldg_pretrain-c6237d5b.pth'
persistent_workers = False

# model settings
model = dict(type='SimpleYOLOWorldDetector',
             mm_neck=True,
             num_train_classes=num_training_classes,
             num_test_classes=num_classes,
             embedding_path=f'/home/xx/YOLO-World/data/texts/{dataset_name}_embeddings.npy',
             prompt_dim=text_channels,
             num_prompts=num_classes,  
             freeze_prompt=False,
             data_preprocessor=dict(type='YOLOv5DetDataPreprocessor'),
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
coco_train_dataset = dict(
        _delete_=True,
        type='MultiModalDataset',
        dataset=dict(
            type='YOLOv5CocoDataset',
            metainfo=dict(classes=class_names),
            data_root=f'data/{dataset_name}',
            ann_file=f'Annotations/{dataset_name}_train.json',
            data_prefix=dict(img='train2017/'),
            filter_cfg=dict(filter_empty_gt=False, min_size=32)),
        class_text_path=f'data/texts/{dataset_name}_text.json',
        pipeline=_base_.train_pipeline)

train_dataloader = dict(persistent_workers=persistent_workers,
                        batch_size=train_batch_size_per_gpu,
                        collate_fn=dict(type='yolow_collate'),
                        dataset=coco_train_dataset)
coco_val_dataset = dict(
                        _delete_=True,
                        type='MultiModalDataset',
                        dataset=dict(
                            type='YOLOv5CocoDataset',
                            data_root=f'data/{dataset_name}',
                            metainfo=dict(classes=class_names),
                            ann_file=f'Annotations/{dataset_name}_test.json',
                            data_prefix=dict(img='val2017/'),
                            filter_cfg=dict(filter_empty_gt=False, min_size=32)),
                        class_text_path=f'data/texts/{dataset_name}_text.json',
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
val_evaluator = dict(
    _delete_=True,
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file=f'data/{dataset_name}/Annotations/{dataset_name}_test.json',
    metric='bbox')
# add evaluator
test_evaluator=val_evaluator    
outfile_prefix= 'train_result'
vis_backends=[dict(type='TensorboardVisBackend')]
visualizer = dict(
    name='visualizer',
    type='mmdet.DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend')
    ])
find_unused_parameters = True

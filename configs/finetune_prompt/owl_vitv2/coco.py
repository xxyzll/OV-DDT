_base_ = ('../../../third_party/mmyolo/configs/yolov8/'
          'yolov8_s_syncbn_fast_8xb16-500e_coco.py')
custom_imports = dict(imports=['yolo_world', 'owl_vitv2'], allow_failed_imports=False)

# hyper-parameters
img_scale = (940, 940)
num_classes = 80
num_training_classes = 80
max_epochs = 80  # Maximum training epochs
close_mosaic_epochs = 10
save_epoch_intervals = 5
text_channels = 512
neck_embed_channels = [128, 256, _base_.last_stage_out_channels // 2]
neck_num_heads = [4, 8, _base_.last_stage_out_channels // 2 // 32]
base_lr = 2e-4
weight_decay = 0.05
train_batch_size_per_gpu = 8
load_from = '/home/xx/YOLO-World/pretrained_models/yolo_world_v2_s_obj365v1_goldg_pretrain-55b943ea.pth'
persistent_workers = False

# model settings
model = dict(
        type='OWL_VitV2',
        cfg='google/owlv2-base-patch16-ensemble',
        texts='data/texts/coco_class_texts.json'
    )

train_pipline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape')),
    
]

val_pipline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape'))
]

# dataset settings
coco_train_dataset = dict(type='YOLOv5CocoDataset',
                          data_root='data/coco',
                          ann_file='annotations/instances_train2017.json',
                          data_prefix=dict(img='train2017/'),
                          filter_cfg=dict(filter_empty_gt=False, min_size=32),
                          pipeline=train_pipline)

train_dataloader = dict(persistent_workers=persistent_workers,
                        batch_size=train_batch_size_per_gpu,
                        collate_fn=dict(type='yolow_collate'),
                        dataset=coco_train_dataset)

coco_val_dataset = dict(type='YOLOv5CocoDataset',
                        data_root='data/coco',
                        ann_file='annotations/instances_val2017.json',
                        data_prefix=dict(img='val2017/'),
                        filter_cfg=dict(filter_empty_gt=False, min_size=32),
                        pipeline=val_pipline)

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
]
train_cfg = dict(max_epochs=max_epochs,
                 val_interval=5,
                 dynamic_intervals=[((max_epochs - close_mosaic_epochs),
                                     5)])

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
# find_unused_parameters = True
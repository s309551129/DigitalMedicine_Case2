_base_ = [
    './_base_/models/swin_transformer/base_224.py',
    './_base_/datasets/imagenet_bs64_swin_224.py',
    './_base_/schedules/imagenet_bs1024_adamw_swin.py',
    './_base_/default_runtime.py'
]

# model setting
model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin_base_patch4_window7_224_22kto1k-f967f799.pth',
            prefix='backbone',
        )),
    head=dict(num_classes=3),
    train_cfg=dict(augments=[
        dict(type='BatchMixup', alpha=0.8, num_classes=3, prob=0.5),
        dict(type='BatchCutMix', alpha=1.0, num_classes=3, prob=0.5)
    ])
)

# dataset setting
data = dict(
    samples_per_gpu=32,
    train = dict(
    type='Filelist',
    data_prefix='../data/train_images',
    ann_file = '../data/train_labels.txt',
    ),
    val = dict(
    type='Filelist',
    data_prefix='../data/train_images',
    ann_file = '../data/valid_labels.txt',
    ),
    test=dict(
    type='Filelist',
    data_prefix='../data/valid_images',
    ann_file = '../data/sample_labels.txt',
    ),
)

evaluation = dict(interval=1, metric='f1_score')
log_config = dict(interval=10, hooks=[dict(type='TextLoggerHook')])
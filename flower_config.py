_base_ = [
    'mmpretrain/configs/_base_/models/resnet50.py',
    'mmpretrain/configs/_base_/datasets/imagenet_bs32.py',
    'mmpretrain/configs/_base_/schedules/imagenet_bs256.py',
    'mmpretrain/configs/_base_/default_runtime.py'
]

# 修改模型头部
model = dict(
    head=dict(
        num_classes=5,
        topk=(1,),
    ))

# 修改数据集配置
data_root = 'data/organized_flower_dataset/'
train_dataloader = dict(
    dataset=dict(
        ann_file='train.txt',
        data_root=data_root,
        data_prefix='train/',
    ))
val_dataloader = dict(
    dataset=dict(
        ann_file='val.txt',
        data_root=data_root,
        data_prefix='val/',
    ))
test_dataloader = val_dataloader

# 修改评估器
val_evaluator = dict(type='Accuracy', topk=(1,))
test_evaluator = val_evaluator

# 修改学习率
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001))

# 减少训练周期
train_cfg = dict(by_epoch=True, max_epochs=20, val_interval=1)

# 预训练模型
load_from = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth'

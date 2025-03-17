_base_ = "base_dino_detr_ssod_coco_r101.py"

data = dict(
    samples_per_gpu=5,
    workers_per_gpu=5,
    train=dict(
        sup=dict(
            type="CocoDataset",
            ann_file="/netscratch/shehzadi/All-Datas/coco/original/annotations/coco_10fully90Unsup.json",
            img_prefix="/netscratch/shehzadi/All-Datas/coco/original/train2017/",

        ),
        unsup=dict(
            type="CocoDataset",
            ann_file="/netscratch/shehzadi/All-Datas/coco/original/annotations/coco_unlabel_10fully90Unsup.json",
            img_prefix="/netscratch/shehzadi/All-Datas/coco/original/train2017/",

        ),
    ),
    val=dict(
            type="CocoDataset",
            ann_file="/netscratch/sarode/coco/original/annotations/instances_val2017.json",
            img_prefix="/netscratch/sarode/coco/original/val2017",

        ),
    test=dict(
            type="CocoDataset",
            ann_file="/netscratch/sarode/coco/original/annotations/instances_val2017.json",
            img_prefix="/netscratch/sarode/coco/original/val2017",

        ),
    sampler=dict(
        train=dict(
            sample_ratio=[1, 4],
        )
    ),
)

semi_wrapper = dict(
    type="DinoDetrSSOD",
    model="${model}",
    train_cfg=dict(
        use_teacher_proposal=False,
        pseudo_label_initial_score_thr=0.4,
        min_pseduo_box_size=0,
        unsup_weight=4.0,
        aug_query=False,
        
    ),
    test_cfg=dict(inference_on="student"),
)

custom_hooks = [
    dict(type="NumClassCheckHook"),
    dict(type="MeanTeacher", momentum=0.999, interval=1, warm_up=0),
    dict(type='StepRecord', normalize=False),
]

runner = dict(_delete_=True, type="IterBasedRunner", max_iters=120000)

fold = 1
percent = 1

work_dir = "work_dirs/${cfg_name}/${percent}/${fold}"
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type='TensorboardLoggerHook')
    ],
)

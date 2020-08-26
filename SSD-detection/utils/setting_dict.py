

setting_dict = {
    "model" :
{
    "backbone" : "mobileNet-V3",
    "boxhead"  :
    {
        "name": "SSD-boxhead",
        "setting":
        {
            "center_var" : 0.1,
            "size_var"  : 0.2,
            "Processor" :
                {
                    #"WIDTH"
                },
            "prior_box" :
                {
                    "name" : "PriorBox",
                    "setting" :
                        {
                            "FEATURE_MAPS" : [32, 16 ,8,4,2,1],
                            "MAX_SIZES" : [40, 148, 236, 324, 412, 500],
                            "MIN_SIZES" : [16,40,148,236,324, 412],
                            "STRIDES" : [16,32,64, 128, 256, 512],
                            # "FEATURE_MAPS" : [32, 16 ,8,4,2,1],
                            # "MAX_SIZES" : [110, 190, 270, 350, 430, 510],
                            # "MIN_SIZES" : [30,110,190,270,350,430 ],
                            # "STRIDES" : [25,32,64, 128, 256, 512],
                            "ASPIECT_RATIOS" : [[], [2,3], [2, 1.4], [2, 1.4], [], []],
                            "CLIP" : True
                        }
                },
            "boxloss" :
                {
                    "name" : "MutiBoxLoss",
                    "setting" :
                        {
                            "ratio" : 3,
                            "class_num"  : 5  #fine_tune 5, # pretrain 32
                        }
                },
            "predictor" :
                {
                    "name" : "BoxPredict-SSD",
                    "setting" :
                        {
                            "class_num" : 5, # fine_tune 5 # pretrain 32
                            "out_channels" : [128,128,128,128,128,128],
                            "box_num"  : [2,6,6,6,2,2]
                        }
                }
        }
    },
},
    "device" : "cuda",
    "solver" :
    {
        "LR" : 0.001,
        "LR_STEP" : [80000, 100000],
        "LRscheduler" :"WarmUpScheduler",
        "optimizer" :
            {
                "momentum" : 0.9,
                "weight_decay" : 0.0005
            }
    },
        "test" : {
            "data_set" : ["/DJI/DJItest"],
            "batch_size" : 2,
            "transform" :
                {
                    "PIXEL_MEAN" : [123, 117, 104],
                    "IMAGE_SIZE" : 512,
                }
        },
        "train": {
            "data_set" :  ["/DJI/DJItrain/"],
            "batch_size" : 8,
            "transform" :
                {
                    "PIXEL_MEAN" : [123, 117, 104],
                    "IMAGE_SIZE" : 512,
                },
        },
    "out_dir" : "/out_dir",
    "train_epoch" : 2000
}


class_name=('__background__',
                'car', 'RedArm1','RedArm2', 'BlueArm1', 'BuleArm2')   # finetune
                # 'Car', 'BlueArmor4', 'RedArmor1', 'RedArmor3', 'RedArmor2', 'RedArmor5', 'Watcher', 'BlueArmor7',
                # 'BlueArmor5', 'BlueArmor2', 'RedArmorNone', 'RedArmor8', 'RedArmor6', 'GreyArmor5', 'GreyArmor4',
                # 'GreyArmorNone', 'BlueArmor1', 'Base', 'GreyArmor7', 'BlueArmor8', 'BlueArmor6', 'GreyArmor3',
                # 'GreyArmor2', 'BlueArmorNone', 'Ignore', 'BlueArmor3', 'RedArmor4', 'GreyArmor1', 'RedArmor7',
                # 'GreyArmor8', 'GreyArmor6' ) # pretrain



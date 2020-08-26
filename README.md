# ICRA-RoboMaster-2020-Perception
 ICRA2020 AI Challenge Northwestern Polytechnical University FireFly Team Perception Code Repository 



## using SSD-detection



### train

* set your dataset(voc) path which contains images, annotation directory in ./utils/setting_dict.py

```python
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
```

* run the train.py  

  ```bash
  python3  train.py 
  ```

*  (optional) specify the output path

  your can view the loss using tensorboard if you like , the loss will be store in output directory  

  ```bash
  python3 train.py  --out_dir {YOUR_PATH}
  ```

### test

```bash
python3 test.py 
```

### finetune

```python
python3 train.py --fine_tune 1  --pretrained_model "{PRETRAINED_MODEL_PATH}"
```

## using RTS-deploy

```bash
rosrun  ICRA-vision  ICRA_vision
```


from ssd_detector import  SSDdetector
import  torch
from solver.trainer import do_evaluate
from data.dataset import make_dataLoader
from utils.setting_dict import setting_dict, class_name
from ssd_detector import SSDdetector

import cv2
import numpy
import os






if __name__ == "__main__" :
    checkpoint = torch.load("./out_dir/mobile-v3-ssd-pafpn.pth", map_location=torch.device("cpu"))
    model = SSDdetector(setting_dict=setting_dict["model"])
    model.load_state_dict(checkpoint.pop("model"))
    #
    # datLoader = make_dataLoader(setting_dict["test"],False)
    # model.eval()
    #
    # do_evaluate(model, datLoader, torch.device("cpu"), "./", 0)
    #
    cap = cv2.VideoCapture("./VID_20200818_100056.mp4")
    ## 16 位 推理
    model = model.cuda().half()
    model.eval()
    writer = cv2.VideoWriter()
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer.open("./result.mp4",fourcc,30,(1920,1080))
    while True :
        _, image_ = cap.read()

        image = cv2.resize(image_,(512,512))
        image_input = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_input =  image_input -  numpy.array((123,117,104))

        image_input = torch.from_numpy(image_input)[None,...]
        image_input = torch.as_tensor(image_input,dtype=torch.float32)

        image_input = image_input.permute(0,3,1,2).contiguous()

        image_input = image_input.cuda().half()
        import time
        start  = time.time()
        detections = model(image_input)
        #torch.onnx.export(model, image_input,"model.onnx",opset_version=10,export_params=True)
        end = time.time()

        print("cost time :{} s".format(end - start))

        bbox  = (detections[0]["boxes"].cpu().detach().numpy())
        score = (detections[0]["scores"].cpu().detach().numpy())
        index = (detections[0]["indexs"].cpu().numpy())
        for i in range(bbox.shape[0]) :
            box = (bbox[i])
            box[0::2] = box[0::2] * 1920
            box[1::2] = box[1::2] * 1080
            if score[i] < 0.45:
                continue
            cv2.rectangle(image_,(box[0],box[1]),(box[2],box[3]),(255,255,255), 2)
            cv2.putText(image_, str(class_name[index[i]]),(box[0], box[1]),1,0.02*(box[2]-box[0]),(255,255,255))
        writer.write(image_)
        cv2.imshow("test", image_)
        cv2.waitKey(1)
    writer.release()



/*************************************************
COPYRIGHT(C),DJI RoboMaster 2019 ICRA AI Challenge,NWPU Firefly Team, Ltd.
FILE NAME: Det.cpp
Author: NWPU Firefly Team
Version: V1.0
Date: 2020/8/23 下午4:22
Email: "wpkkstr@gmail.com"
DESCRIPTION: This file defines the function body of superclass related
 functions for detecting and tracking robots.
*************************************************/

#include "Det.h"
#include <opencv2/tracking.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utility.hpp>

torch::Tensor cv2tensor(cv::Mat input)
{
  cv::cvtColor(input,input,cv::COLOR_BGR2RGB);
  cv::resize(input,input,cv::Size(416,256));

  input.convertTo(input,CV_32FC3,1.0/255.0f);

  return torch::from_blob(input.data,{256,416,3}).unsqueeze(0).permute({0,3,1,2}).contiguous();
}

// set tracker

void Det::setTracker()
{
  //trackerPtr = cv::TrackerKCF::create();
  tracker = cv::MultiTracker::create();
}


//set Net
void Det::setNet(std::string netPath)
{
  Net = torch::jit::load(netPath);
  Net.eval();

  //Net.to(at::DeviceTypeName(c10::DeviceType::CUDA));
}

void Det::setTrackNumThreshold(int _num)
{
  trackNumThreshold = _num;
}

// set nms
void Det::setNms(float _nmsThreshold)
{
  this->nmsThreshold = _nmsThreshold;
}

// set pre
void Det::setPre(float _preThreshold)
{
  this->preThreshold = _preThreshold;
}

void Det::setDetNum(int _detNum)
{
  this->detectNumThreshold = _detNum;
}

void Det::setTrackThreshold(int _trackThreshold)
{
  this->trackThreshold = _trackThreshold;
}

// set input
void Det::setInput(cv::Mat input)
{
  frame = input.clone();

  //std::cout<< frame.size <<std::endl;

  netInput = cv2tensor(input);
}

void Det::resetTracker()
{
  trackNum = 0;
  //trackerBool.clear();
  Rects.clear();
  tracker->clear();
  cv::Ptr<cv::MultiTracker> newTracker = cv::MultiTracker::create();
  tracker = newTracker;
  //std::cout<<windowsV.size()<<std::endl;
  for(int i=0;i<windowsV.size();i++)
  {
    Rects.push_back(windowsV[i].Roi);

    cv::Ptr<cv::Tracker> temp = cv::TrackerKCF::create();
    bool success = tracker->add(temp,frame,windowsV[i].Roi);
    //std::cout<<"bool:"+std::to_string(success)<<std::endl;
  }
  //std::cout<<"set finish"<<std::endl;
  state = RunState::Tracking;
}

//sleep and reset
void Det::sleep(int _time)
{

}

// track
void Det::track()
{
  trackNum++;
  tracker->update(frame,Rects);
  for(int i=0;i<windowsV.size();i++)
  {
    windowsV[i].update(Rects[i]);
  }

  if(trackNum < trackNumThreshold)
  {
    state = RunState::Tracking;
  } else{
    state = RunState::Detecting;
  }
}

// detect
void Det::detect()
{
  netResult = Net.forward({netInput}).toTuple()->elements()[0].toTensor();

  netResult = netResult.squeeze(0);
  //std::cout<<netResult.sizes()<<std::endl;

  nmsSuppression();

  if(windowsV.size() > 0)
  {
    detectNum++;
  } else{
    detectNum = 0;
  }
  if(detectNum > detectNumThreshold && windowsV.size() <= trackThreshold)
  {
    state = RunState::toTrack;
    detectNum = 0;
  } else{
    state  = RunState::Detecting;
  }
}

// nms
void Det::nmsSuppression()
{
  windowsV.clear();
  netResult.slice(1,5,12) *= netResult.slice(1,4,5);
  std::tuple<torch::Tensor,torch::Tensor> vL= (netResult.slice(1,5).max(1));
  torch::Tensor maxPreV = std::get<0>(vL);
  torch::Tensor maxLocV = std::get<1>(vL);
  torch::Tensor netslice = netResult.slice(1,4,5);
//    std::cout<<maxPreV.max()<<std::endl;
//    std::cout<<netResult.max()<<std::endl;
//    std::cout<<netResult[0]<<std::endl;
  //netslice*=maxPreV;
  //netResult = torch::stack(torch::TensorList({netResult.slice(1,0,5),maxLocV}));
  //std::cout<<netResult.sizes()<<std::endl;
  int length = netResult.size(0);

  for(int i=0;i<length;i++)
  {
    if(netResult[i][4].item().toFloat() < preThreshold)continue;
    if(netResult[i][2].item().toFloat() < 2)continue;
    if(netResult[i][3].item().toFloat() < 2)continue;
    if(netResult[i][2].item().toFloat() > 3000)continue;
    if(netResult[i][3].item().toFloat() > 3000)continue;


    int index_ = maxLocV[i].item().toInt();
    window win(netResult[i][0].item().toFloat(),
               netResult[i][1].item().toFloat(),
               netResult[i][2].item().toFloat(),
               netResult[i][3].item().toFloat(),
               index_);
    win.score = netResult[i][4].item().toFloat()*maxPreV[i].item().toFloat();
    bool flag = 0;
    for(int j=0 ;j<windowsV.size();j++)
    {
      if (index_ != windowsV[j].label)
      {
        continue;
      }
      float iou = IOU(windowsV[j], win);
      if (iou > nmsThreshold)
      {
        flag = 1;
        if (win.score > windowsV[j].score)
        {
          windowsV[j] = win;
        }
        else{

        }
        break;

      } else{

      }
    }
    if(flag == 0)
    {
      windowsV.push_back(win);
      //std::cout<<win.score<<std::endl;
    }
  }
}

Det::Det(std::string netPath):tracker(cv::MultiTracker::create())
{
  num++;
  setNet(netPath);
  setPre(0.3);
  setNms(0.5);
  setDetNum(99999);
  setTrackThreshold(5);
  setTrackNumThreshold(8);
  state = RunState::Detecting;
  label= num;
}

std::vector<window> & Det::getWindows()
{
  return windowsV;
}
void Det::show()
{
  cv::Mat display_src = frame.clone();
  cv::resize(display_src,display_src,cv::Size(416,256));
  std::cout<<"detected"<<std::endl;
  for( auto window : windowsV)
  {
    //std::cout<<"1"<<std::endl;
    cv::rectangle(display_src,window.Roi,cv::Scalar::all(int(window.label/7.0*255.0f)));
    int baseLine;
    cv::Size labelSize = getTextSize("class index:"+std::to_string(window.label), cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    cv::rectangle(display_src,cv::Point(window.x1,window.y1-labelSize.height),
                  cv::Point(window.x1 + labelSize.width,window.y1+baseLine),
                  cv::Scalar::all(100),-1);
    cv::putText(display_src,"class index:"+std::to_string(window.label),cv::Point(window.x1,window.y1),
                cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(255,255,255));
    //break;
  }
  //std::cout<<1.5<<std::endl;
  //cv::namedWindow("DETECTION");
  //std::cout<<"2"<<std::endl;
  // cv::imshow("eryhcu" + std::to_string(label),display_src);
  //std::cout<<3<<std::endl;
  cv::waitKey(10);
  //std::cout<<4<<std::endl;
}



void Det::stateMachine()
{
  //std::cout<<"??????????"<<std::endl;
  switch (state)
  {
  case RunState::Detecting :
  {
    detect();
    break;
  }
  case RunState::Tracking :
  {
    track();
    break;
  }
  case RunState::toTrack:
  {
    resetTracker();
    track();
    break;
  }
  case RunState::Waiting:
  {
    sleep(10);
  }
  }
  // 可关闭
  show();
}
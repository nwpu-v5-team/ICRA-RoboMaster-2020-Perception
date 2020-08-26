/*************************************************
COPYRIGHT(C),DJI RoboMaster 2019 ICRA AI Challenge,NWPU Firefly Team, Ltd.
FILE NAME: Det.h
Author: NWPU Firefly Team
Version: V1.0
Date: 2020/8/23 下午4:24
Email: "wpkkstr@gmail.com"
DESCRIPTION: This file defines the superclass of detecting and tracking
 robots, including the basic members and basic methods of the class.
*************************************************/

#ifndef UNTITLED_DET_H
#define UNTITLED_DET_H

#include "Window.h"
#include <string>
#include <torch/torch.h>
#include <torch/jit.h>
#include <torch/script.h>
//#include <torch>
#include <opencv2/tracking.hpp>
#include <opencv2/opencv.hpp>
#include "State.h"
typedef cv::Ptr<cv::Tracker> TrackerPtr;

class Det
{
  static int num;
  int label;
public:
  Det(){};

  explicit Det(std::string  netPath);

  void setPre(float _preThreshold);

  void setNms(float _nmsThreshold);

  void setInput(cv::Mat input);

  void setNet(std::string netPath);

  void setTrackNumThreshold(int _num);

  void setTrackThreshold(int _trackThreshold);

  void setDetNum(int _detNum);

  void resetTracker();

  void setTracker();

  void nmsSuppression();

  void track();

  void detect();

  void sleep(int _time);

  void stateMachine();

  std::vector<window> &getWindows();

private:
  float preThreshold;
  float nmsThreshold;

  int trackThreshold;
  int trackNum;
  int trackNumThreshold;
  int detectNumThreshold;
  int detectNum;
  cv::Mat frame;

  torch::jit::script::Module Net;

  torch::Tensor netResult;

  torch::Tensor netInput;

  std::vector<window> windowsV;

  cv::Ptr<cv::MultiTracker> tracker;

  std::vector<bool> trackerBool;

  std::vector<cv::Rect2d> Rects;
private:
  RunState state;

  void show();
};

#endif //UNTITLED_DET_H


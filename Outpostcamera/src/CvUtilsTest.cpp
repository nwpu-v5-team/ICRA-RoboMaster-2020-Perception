/*************************************************
COPYRIGHT(C),DJI RoboMaster 2019 ICRA AI Challenge,NWPU Firefly Team, Ltd.
FILE NAME: CvUtilTest.cpp
Author: NWPU Firefly Team
Version: V1.0
Date: 2020/8/23 下午4:35
Email: "wpkkstr@gmail.com"
DESCRIPTION:This file defines the test function of the visualization function.
*************************************************/

#include "CvUtils.h"
std::vector<cv::Point> HmatrixHelper::pixPointV_temp = {};
int main()
{
  HmatrixHelper testNode;
  cv::VideoCapture capture(0);
  cv::Mat testFrame;
  //capture>>testFrame;
  //testNode.setFrame(testFrame);
  //testNode.setPixPointV();
  testNode.setRealPointV("/home/wang_shuai/points.yaml");
}

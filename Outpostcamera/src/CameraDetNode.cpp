/*************************************************
COPYRIGHT(C),DJI RoboMaster 2019 ICRA AI Challenge,NWPU Firefly Team, Ltd.
FILE NAME: CameraDetNode.cpp
Author: NWPU Firefly Team
Version: V1.0
Date: 2020/8/23 下午4:17
Email: "wpkkstr@gmail.com"
DESCRIPTION: This file defines the entry function
 of the entire program and implements the main
 loop of the ros program.
Function List: main()
*************************************************/

#include "RosDet.h"
#include <ros/package.h>
#include <ros/ros.h>

std::vector<cv::Point> HmatrixHelper::pixPointV_temp = {};

int Det::num = 0;

int main(int argc,char** argv)
{
  ros::init(argc,argv,"cameraGcNode");

  YAML::Node config = YAML::LoadFile(ros::package::getPath("robot_vision") +"/tools/config.yaml");

  rosDet worker(config);

  worker.advertiseFunction();

  //ros::spinOnce();

  ros::spin();
}

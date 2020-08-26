/*************************************************
COPYRIGHT(C),DJI RoboMaster 2019 ICRA AI Challenge,NWPU Firefly Team, Ltd.
FILE NAME: CvUtils.h
Author: NWPU Firefly Team
Version: V1.0
Date: 2020/8/23 下午4:33
Email: "wpkkstr@gmail.com"
DESCRIPTION: This file defines auxiliary functions that use OpenCV and callback
 functions to realize visualization functions.
*************************************************/

#ifndef ROBOT_VISION_CVUTILS_H
#define ROBOT_VISION_CVUTILS_H

#include <opencv2/opencv.hpp>
#include <ros/package.h>
#include <ros/ros.h>


class HmatrixHelper
{
private:
  cv::Mat frame;
  std::vector<cv::Point> realPointV;
  std::vector<cv::Point> pixPointV;

public :
  cv::Mat homographyMatrix;

public:

  static std::vector<cv::Point> pixPointV_temp;

  static void mouseCallBack(int event, int x, int y, int flags, void *param);

  static void flush(cv::Mat frame);

  HmatrixHelper(){};

  void setRealPointV(std::string path);

  void setPixPointV();

  void computeHM();

  void setFrame(cv::Mat frame);

};

class visualHelper
{
private:
  cv::Mat frameX;

  std::vector<cv::Point> roboPosV;

public:
  visualHelper()
  {
    cv::Mat framex(488,848,CV_8UC3,cv::Scalar(0,0,0));
    cv::Mat exchange = cv::imread(ros::package::getPath("robot_vision")+"/photo/icra_ground_plane.png");
    // cv::imshow("ex",exchange);
    // cv::waitKey(1);
    exchange.copyTo(framex);
    // frame += cv::Scalar(200,200,80);
    cv::Mat frame = framex(cv::Rect(20,20,808,448));
    cv::rectangle(frame,cv::Point(0,448-348),cv::Point(100,448-328),cv::Scalar(228,221,220),-1);//b1
    cv::rectangle(frame,cv::Point(808-354-100,448-308),cv::Point(808-354,448-328),cv::Scalar(228,221,220),-1);
    cv::rectangle(frame,cv::Point(808-150-20,0),cv::Point(808-150,100),cv::Scalar(228,221,220),-1);
    cv::rectangle(frame,cv::Point(150,448-214-20),cv::Point(230,448-214),cv::Scalar(228,221,220),-1);
    cv::rectangle(frame,cv::Point(808-230,448-214-20),cv::Point(808-150,448-214),cv::Scalar(228,221,220),-1);
    cv::rectangle(frame,cv::Point(808-354-100,308),cv::Point(808-354,328),cv::Scalar(228,221,220),-1);
    cv::rectangle(frame,cv::Point(150,348),cv::Point(170,448),cv::Scalar(228,221,220),-1);
    cv::rectangle(frame,cv::Point(708,328),cv::Point(808,348),cv::Scalar(228,221,220),-1);//b9

    std::vector<cv::Point> ps={{404-17,224},{404,224-17},{404+17,224},{404,224+17}};
    std::vector<std::vector<cv::Point> > pss = {ps};
    cv::drawContours(frame,pss,0,cv::Scalar::all(100),-1);

    // cv::imshow("frame",framex);
    // cv::waitKey();
    cv::imwrite("icra_gp_withob.png",framex);
    frameX = framex.clone();
  }

  void visualize();

  void flush();

  void setRobPosV(std::vector<cv::Point>& pointV);

};

#endif //ROBOT_VISION_CVUTILS_H

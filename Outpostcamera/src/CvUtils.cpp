/*************************************************
COPYRIGHT(C),DJI RoboMaster 2019 ICRA AI Challenge,NWPU Firefly Team, Ltd.
FILE NAME: CvUtils.cpp
Author: NWPU Firefly Team
Version: V1.0
Date: 2020/8/23 下午4:33
Email: "wpkkstr@gmail.com"
DESCRIPTION: This document defines the class for visualization using OpenCV.
 The class mainly implements the function of drawing a picture of the field
 reduced by a scale of 1:10, and the function of displaying the corresponding
 position of the detected robot in the field picture.
*************************************************/

#include "CvUtils.h"
#include <yaml-cpp/yaml.h>

void HmatrixHelper::mouseCallBack(int event, int x, int y, int flags, void *param)
{

  //std::cout<<"test"<<std::endl;
  if (event == cv::EVENT_LBUTTONDOWN)
  {
    std::cout << "添加一个元素" << std::endl;
    pixPointV_temp.push_back(cv::Point(x, y));
    std::cout << "mouse_click_point: " << cv::Point(x,y)<< std::endl;

  } else if (event == cv::EVENT_RBUTTONDOWN)
  {
    if (pixPointV_temp.size()> 0)
    {
      for(std::vector<cv::Point>::iterator ite = pixPointV_temp.begin(); ite!=pixPointV_temp.end();ite++)
      {
        if(cv::abs(ite->x-x) < 10 && cv::abs(ite->y -y ) < 10)
        {
          std::cout << "删除此元素" << std::endl;
          pixPointV_temp.erase(ite);
          break;
        }
      }

    } else
    {
      std::cout<< "what are u fucking doing"<<std::endl;
    }
  }

  //std::cout<<pixPointV_temp<<std::endl;
}

void HmatrixHelper::flush(cv::Mat frame)
{
  cv::Mat _frame = frame.clone();

  for(int i=0;i<pixPointV_temp.size();i++)
  {
    cv::circle(_frame,pixPointV_temp[i],5,cv::Scalar(255,0,0),1);
    cv::circle(_frame,pixPointV_temp[i],2,cv::Scalar(255,0,0),-1);
  }
  cv::imshow("pixPointWindow",_frame);
}

void HmatrixHelper::setFrame(cv::Mat frame)
{
  this->frame = frame.clone();
}

void HmatrixHelper::setRealPointV(std::string path)
{
  realPointV.clear();
  //std::cout<<path<<std::endl;
  //read from file
  YAML::Node config = YAML::LoadFile(path);
  //cv::FileStorage fileNode = cv::FileStorage(path,cv::FileStorage::READ);
  //std::cout<<path<<std::endl;
  int pointNum = config["pointNum"].as<int>();
//    cv::Mat pointM;
//    fileNode["pointM"] >> pointM;
//    std::cout<<path<<std::endl;
  for(int i=0;i<pointNum;i++)
  {
    realPointV.push_back(cv::Point2f(config["pointX"][i].as<float>(),config["pointY"][i].as<float>()));
  }
  //std::cout<<realPointV<<std::endl;
}

void HmatrixHelper::setPixPointV()
{
  cv::namedWindow("pixPointWindow");
  cv::setMouseCallback("pixPointWindow",HmatrixHelper::mouseCallBack);

  pixPointV_temp.clear();

  while(cv::waitKey(10)!= 'q')
  {
    flush(this->frame);
  }

  cv::destroyWindow("pixPointWindow");

  //std::cout<<pixPointV_temp<<std::endl;

  this->pixPointV = pixPointV_temp;
}

void HmatrixHelper::computeHM()
{
  this->homographyMatrix = cv::findHomography(pixPointV,realPointV,cv::RANSAC,2);

  this->homographyMatrix.convertTo(homographyMatrix,CV_32F);

  std::cout<<pixPointV<<std::endl;
  std::cout<<realPointV<<std::endl;
  std::cout<<homographyMatrix<<std::endl;
}

void visualHelper::setRobPosV(std::vector<cv::Point> &pointV)
{
  this->roboPosV = pointV;
}

void visualHelper::flush()
{
  cv::Mat _frame = frameX.clone();
  for(int i=0;i<roboPosV.size();i++)
  {
    cv::rectangle(_frame,cv::Rect(429  + roboPosV[i].x,244- roboPosV[i].y,25,25),cv::Scalar(100,200,245),-1);
  }
  cv::imshow("visualizeRobot",_frame);
  cv::waitKey(1);
}

void visualHelper::visualize()
{
}

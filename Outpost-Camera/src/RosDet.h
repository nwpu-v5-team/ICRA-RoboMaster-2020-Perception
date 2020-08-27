/*************************************************
COPYRIGHT(C),DJI RoboMaster 2019 ICRA AI Challenge,NWPU Firefly Team, Ltd.
FILE NAME: RosDet.h
Author: NWPU Firefly Team
Version: V1.0
Date: 2020/8/23 下午4:28
Email: "wpkkstr@gmail.com"
DESCRIPTION: This file defines a subclass of the Det class, which encapsulates
 the relevant attributes and operations of the ROS project, including the basic
 members and member functions of the class.
*************************************************/

#ifndef ROBOT_VISION_ROSDET_H
#define ROBOT_VISION_ROSDET_H

#include "CvUtils.h"
#include "Det.h"
#include "OutpostDetected.h"
#include "roboPos.h"
#include <cv_bridge/cv_bridge.h>
#include <functional>
#include <image_transport/image_transport.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <thread>
#include <yaml-cpp/yaml.h>

typedef std::vector<robot_vision::roboPos> roboPosV;
class rosDet
{
public :
//    static rosDet& getInstance()
//    {
//        static rosDet item = rosDet();
//        return item;
//    };

  image_transport::Subscriber subscriberConstructer(int i,std::string topic, std::string initPath_YAML);

  void subscribeFunction(int i,std::string initPath_YAML,sensor_msgs::ImageConstPtr ptr);

  void advertiseFunction();

  void detectionProcess(int processID);

private :
  ros::NodeHandle _nh;

  image_transport::ImageTransport _it;

  ros::Publisher  _publisher;

  std::vector<image_transport::Subscriber> _subscriberV;

  std::vector<Det> detectionV;

  std::vector<std::thread> threadV;

  //std::vector<std::shared_ptr<std::mutex> > resumeV;

  std::mutex publishMutex;

  std::mutex InitingMutex;

  std::vector<std::unique_ptr<std::atomic_bool> > initV;

  std::vector<std::unique_ptr<std::atomic_bool> > epochFinishV;

  std::vector<cv::Mat> HomoV;

  visualHelper _visualizer;

  std::vector<roboPosV> roboPosVV;

  robot_vision::OutpostDetected message;


  void threadConstruct(int threadNum);

  void publisherConstruct(std::string topic);

  void subscriberConstruct(std::vector<std::string> topics, std::vector<std::string> realPoints);

  void detectionConstruct(std::vector<std::string> dnnPaths);

  cv::Mat getPosAllInOne(int u,int v,cv::Mat K ,cv::Mat Rwc,cv::Mat Twc);


public :
  rosDet(YAML::Node  config):_nh(),_it(_nh)
  {

    int num = config["num"].as<int32_t>();

    detectionV.resize(num);

    threadV.resize(num);

    //resumeV.resize(num);

    HomoV.resize(num);

    epochFinishV.resize(num);

    initV.resize(num);
    epochFinishV.resize(num);


    {
      for(int i=0;i<num;i++)
      {
        initV[i] = std::make_unique<std::atomic_bool>(false);
        epochFinishV[i] = std::make_unique<std::atomic_bool>(false);
      }
    }


    roboPosVV = {{},{}};
    _subscriberV.resize(num);

    std::vector<std::string> subsribeV,detDnnPathV,realPointsV;

    for(int i=0;i<num;i++)
    {
      subsribeV.push_back(config["subscribeTopic"][i].as<std::string>());
      detDnnPathV.push_back(config["detectionDnnPath"][i].as<std::string>());
      realPointsV.push_back(config["realWorldPoints"][i].as<std::string>());
    }

    //std::cout<<subsribeV<<std::endl;
    detectionConstruct(detDnnPathV);

    subscriberConstruct(subsribeV,realPointsV);

    //std::cout<<"fin"<<std::endl;

    publisherConstruct(config["publishTopic"].as<std::string>());

    //std::cout<<"fin2"<<std::endl;

    threadConstruct(num);




//            threadV[0].join();
//            threadV[1].join();
//            threadV[2].join();

    //std::cout<<"888"<<std::endl;
  }

};

#endif //ROBOT_VISION_ROSDET_H

/*************************************************
COPYRIGHT(C),DJI RoboMaster 2019 ICRA AI Challenge,NWPU Firefly Team, Ltd.
FILE NAME: RosDet.cpp
Author: NWPU Firefly Team
Version: V1.0
Date: 2020/8/23 下午4:30
Email: "wpkkstr@gmail.com"
DESCRIPTION: This file implements the relevant member functions of the RosDet class,
 including loading parameters, obtaining current camera information, reading in the image
 and calculating with the neural network, returning the detection result and calculating
 the spatial position information of the robot, encapsulating it into a ROS custom message
 and Publish and pass the results to the visualization module at the same time.
*************************************************/

#include "RosDet.h"

image_transport::Subscriber
rosDet::subscriberConstructer(int i, std::string topic,
                              std::string initPath_YAML) {
  std::function<void(sensor_msgs::ImageConstPtr)> functionBind =
      std::bind(&rosDet::subscribeFunction, this, i, initPath_YAML,
                std::placeholders::_1);

  image_transport::Subscriber subscriber =
      _it.subscribe(topic, 1, functionBind);

  return subscriber;
}

void rosDet::subscribeFunction(int i, std::string initPath_YAML,
                               sensor_msgs::ImageConstPtr ptr) {

  // if(*initV[i] == true)return;
  // std::lock_guard<std::mutex> lockGuard(InitingMutex);
  // std::cout<<"receive"<<std::endl;
  detectionV[i].setInput(cv_bridge::toCvShare(ptr, "bgr8")->image);

  if (*initV[i] == false) {
    {
      /*
      HmatrixHelper _converter;
      _converter.setFrame(cv_bridge::toCvShare(ptr, "bgr8")->image);
      //std::cout<<"setframe"<<std::endl;
      _converter.setRealPointV(initPath_YAML);
      //std::cout<<"setPoints"<<std::endl;
      _converter.setPixPointV();
      //std::cout<<"setPix"<<std::endl;
      //_converter.computeHM();
      HomoV[i] = _converter.homographyMatrix;
      //std::cout<<i<<std::endl;
      */
      *(initV[i]) = true;
      // HomoV[i] = cv::Mat_<float>(3,3);
      // HomoV[i].at<float>(0,0) = -12.4218;
      // HomoV[i].at<float>(0,1) = -6.3978853;
      // HomoV[i].at<float>(0,2) = 9320.1553;
      // HomoV[i].at<float>(1,0) = 7.6027651;
      // HomoV[i].at<float>(1,2) = -1.48577;
      // HomoV[i].at<float>(1,3) = 1536.7446;
      // HomoV[i].at<float>(2,0) = 0;
      // HomoV[i].at<float>(2,1) = 0;
      // HomoV[i].at<float>(2,2) = 1;
    }
  }
}

void rosDet::threadConstruct(int threadNum) {
  for (int i = 0; i < threadNum; i++) {
    threadV[i] = std::thread(&rosDet::detectionProcess, this, i);

    threadV[i].detach();
  }
  // threadV[threadNum] = (std::thread(&rosDet::advertiseFunction,this));
  // threadV[threadNum].detach();
}
void rosDet::detectionConstruct(std::vector<std::string> dnnPaths) {
  // std::cout<<dnnPaths.size()<<std::endl;
  for (int i = 0; i < dnnPaths.size(); i++) {
    detectionV[i] = Det(dnnPaths[i]);
  }
}
/*outpost_camera2四元数
 * rotation:
        x: 0.379515108914
        y: 0.115193930491
        z: -0.0476880228429
        w: 0.916746689638

 * */

cv::Mat rosDet::getPosAllInOne(int u, int v, cv::Mat K, cv::Mat Rwc,
                               cv::Mat Twc) {
  float fx = K.at<float>(0, 0);
  float fy = K.at<float>(1, 1);
  float cx = K.at<float>(0, 2);
  float cy = K.at<float>(1, 2);

  cv::Mat pt_cam = cv::Mat(3, 1, CV_32FC1);
  cv::Mat pt_mid = cv::Mat(3, 1, CV_32FC1);
  cv::Mat pt_wld = cv::Mat(3, 1, CV_32FC1);
  cv::Mat PT_CAM = cv::Mat::zeros(3, 1, CV_32FC1);
  cv::Mat PT_WLD = cv::Mat::zeros(3, 1, CV_32FC1);
  pt_cam.at<float>(2, 0) = 1;                 // z
  pt_cam.at<float>(0, 0) = 1 * (u - cx) / fx; // x
  pt_cam.at<float>(1, 0) = 1 * (v - cy) / fy; // y

  // step 2 :获取gazebo坐标系下的坐标
  pt_mid.at<float>(0, 0) = pt_cam.at<float>(2, 0);  // x' = z
  pt_mid.at<float>(1, 0) = -pt_cam.at<float>(0, 0); // y'  = -x
  pt_mid.at<float>(2, 0) = -pt_cam.at<float>(1, 0); // z' = -y

  // step 3 :获取世界坐标系下的坐标
  pt_wld = Rwc * pt_mid + Twc;
  PT_WLD = Rwc * PT_CAM + Twc;

  // step 4 :已知直线平面，获取投影直线和平面的交点坐标--最终现实坐标系下的坐标
  cv::Mat VL = pt_wld - PT_WLD;
  cv::Mat VP = cv::Mat(3, 1, CV_32FC1);
  VP.at<float>(0, 0) = 0;
  VP.at<float>(1, 0) = 0;
  VP.at<float>(2, 0) = 1;

  float V1 = VL.at<float>(0, 0);
  float V2 = VL.at<float>(1, 0);
  float V3 = VL.at<float>(2, 0);

  float Vp1 = VP.at<float>(0, 0);
  float Vp2 = VP.at<float>(1, 0);
  float Vp3 = VP.at<float>(2, 0);

  float M1 = pt_wld.at<float>(0, 0);
  float M2 = pt_wld.at<float>(1, 0);
  float M3 = pt_wld.at<float>(2, 0);

  const float N1 = 0;
  const float N2 = 0;
  const float N3 = 0.1;

  float T = (N1 - M1) * Vp1 + (N2 - M2) * Vp2 + (N3 - M3) * Vp3;
  if ((Vp1 * V1 + Vp2 * V2 + Vp3 * V3) == 0) {
    std::cout << "divide within 0!" << std::endl;
  }
  T /= Vp1 * V1 + Vp2 * V2 + Vp3 * V3;
  if (T == ((N3 - M3) / V3)) {
    std::cout << "T is as expected" << std::endl;
  }

  cv::Mat target_pos = cv::Mat(3, 1, CV_32FC1);
  target_pos.at<float>(0, 0) = M1 + V1 * T;
  target_pos.at<float>(1, 0) = M2 + V2 * T;
  target_pos.at<float>(2, 0) = N3;

  return target_pos;
}

void rosDet::detectionProcess(int processID) {
  while (1) {
    std::cout << processID << std::endl;
    if (*(initV[processID]) == true) {
      // std::cout<<"here"<<std::endl;

      detectionV[processID].stateMachine();

      std::cout << "finish" << std::endl;
      std::vector<window> windows = detectionV[processID].getWindows();
      roboPosVV[processID].clear();

      {
        std::lock_guard<std::mutex> lock(publishMutex);
        for (int i = 0; i < windows.size(); i++) {
          cv::Mat pos = cv::Mat_<float>(3, 1);
          pos.at<float>(0, 0) = windows[i].getBottomX() / 416.0 * 1280.0;
          pos.at<float>(1, 0) = windows[i].getBottomY() / 256.0 * 720.0;
          pos.at<float>(2, 0) = 1;

          // std::cout<<windows[i].getBottomX()<<std::endl;
          // std::cout<<windows[i].getBottomY()<<std::endl;
          // std::cout<<pos<<std::endl;
          // pos = HomoV[processID]*(pos);
          // std::cout<<pos<<std::endl;
          int u = pos.at<float>(0, 0);
          int v = pos.at<float>(1, 0);
          cv::Mat K = (cv::Mat_<float>(3, 3) << 762.7249337622711, 0.0, 640.5,
              0.0, 762.7249337622711, 360.5, 0.0, 0.0, 1.0);

          std::vector<cv::Mat> Rwc_v;
          Rwc_v.resize(2);
          Rwc_v.clear();
          cv::Mat r1 = (cv::Mat_<float>(3, 3) << -0.684306, 0.707951, -0.174732,
              -0.685942, -0.706262, -0.17515, -0.247404, 2.43625e-13,
              0.968912);
          Rwc_v.push_back(r1);
          cv::Mat r2 = (cv::Mat_<float>(3, 3) << 0.96891242, 0.17487136,
              0.17501065, -1.7258417e-13, 0.70738828, -0.7068252,
              -0.24740396, 0.68485171, 0.68539727);
          /* Eigen::Quaterniond qt(0.916746689638, 0.379515108914,
          0.115193930491,-0.0476880228429);
           Eigen::Matrix3d qt_m = qt.toRotationMatrix();
           for (int row = 0 ; row < 3 ;row++){
             for (int col =0 ;col < 3; col ++){
               r2.at<float>(row,col) = qt_m(row,col);
             }
           }*/
          // std::cout<<r2<<std::endl;
          Rwc_v.push_back(r2);

          std::vector<cv::Mat> Twc_v;
          Twc_v.resize(2);
          Twc_v.clear();
          cv::Mat t1 = (cv::Mat_<float>(3, 1) << 4.04, 2.24, 1.808);
          Twc_v.push_back(t1);
          cv::Mat t2 = (cv::Mat_<float>(3, 1) << -4.04, -2.24, 1.808);
          Twc_v.push_back(t2);

          cv::Mat pos_target = cv::Mat(3, 1, CV_32FC1);
          // pos_target =
          // getPosAllInOne(u,v,K,Rwc_v[processID],Twc_v[processID]);
          pos_target = getPosAllInOne(u, v, K, r1, t1);
          // std::cout<< pos_target <<std::endl;

          robot_vision::roboPos p;
          //                    p.x = pos.at<float>(0, 0);
          //                    p.y = pos.at<float>(1, 0);
          p.x = pos_target.at<float>(0, 0);
          p.y = pos_target.at<float>(1, 0);

          roboPosVV[processID].push_back(p);
          // }
        }
      }
      *(epochFinishV[processID]) = true;
    } else {
      //            ros::Rate sl(30);
      //            sl.sleep();
    }

    // resumeV[processID] = true;
    ros::Rate sll(30);
    sll.sleep();
    std::cout << "oneEPOCH" << std::endl;
  }
}

void rosDet::advertiseFunction() {
  while (1) {
    bool flag = true;
    for (int i = 0; i < 1; i++) {
      flag &= *(epochFinishV[i]);
    }
    std::vector<cv::Point> robotPos_;
    // std::cout<<"*************"<<std::endl;
    if (flag) {
      // std::cout<<"ok"<<std::endl;
      // message.robotPosArray  = robot_vision::roboPos[10];
      {
        std::lock_guard<std::mutex> lock(publishMutex);
        for (int i = 0; i < roboPosVV[0].size(); i++) {
          message.robotPosArray.push_back(roboPosVV[0][i]);
          robotPos_.push_back(
              cv::Point(roboPosVV[0][i].x * 100, roboPosVV[0][i].y * 100));
        }
        // for (int i = 0; i < roboPosVV[1].size(); i++) {
        //     message.robotPosArray.push_back(roboPosVV[1][i]);
        //     robotPos_.push_back(cv::Point(roboPosVV[1][i].x*100,roboPosVV[1][i].y*100));
        // }
        roboPosVV[0].clear();
        // roboPosVV[1].clear();
      }

      _publisher.publish(message);

      for (int i = 0; i < 1; i++) {
        *(epochFinishV[i]) = false;
      }
      _visualizer.setRobPosV(robotPos_);
      _visualizer.flush();
    } else {
    }
    ros::Rate r(100);
    r.sleep();
    ros::spinOnce();
  }
}

void rosDet::subscriberConstruct(std::vector<std::string> topics,
                                 std::vector<std::string> realPoints) {
  for (int i = 0; i < topics.size(); i++) {
    _subscriberV[i] = subscriberConstructer(i, topics[i], realPoints[i]);
  }
}

void rosDet::publisherConstruct(std::string topic) {
  _publisher = _nh.advertise<robot_vision::OutpostDetected>(topic, 1, false);
}

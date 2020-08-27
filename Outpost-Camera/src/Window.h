/*************************************************
COPYRIGHT(C),DJI RoboMaster 2019 ICRA AI Challenge,NWPU Firefly Team, Ltd.
FILE NAME: Window.h
Author: NWPU Firefly Team
Version: V1.0
Date: 2020/8/23 下午4:26
Email: "wpkkstr@gmail.com"
DESCRIPTION: This file defines the class of two-dimensional boxes containing
 successfully detected robots.
*************************************************/

#ifndef UNTITLED_WINDOW_H
#define UNTITLED_WINDOW_H

#include <algorithm>
#include <opencv2/core/types.hpp>

struct window
{
  float x1,y1;
  float x2,y2;
  float centor_x;
  float centor_y;
  float width;
  float height;
  float score;

  cv::Rect2d  Roi;

  int label;
  window(){};
  window(float _centor_x,float _centor_y,float _width,float _height,int _label=-1)
  {
//    { centor_x = _centor_x;
//        centor_y = _centor_y;
//        width = _width;
//        height = _height;
//        x1 = centor_x-width/2;
//        x2 = centor_x+width/2;
//        y1 = centor_y-height/2;
//        y2 = centor_y+height/2;
    centor_x = (_centor_x);
    centor_y = _centor_y;
    width = _width;
    height = _height;
    x1 = (centor_x-width/2);
    x2 = (centor_x+width/2);
    y1 = (centor_y-height/2);
    y2 = (centor_y+height/2);
    label = _label;
    Roi = cv::Rect2d(cv::Point2f(x1,y1),cv::Point2f(x2,y2))&cv::Rect2d(0,0,416,256);
  }

  void update(float _centor_x,float _centor_y,float _width,float _height)
  {
    centor_x = _centor_x;
    centor_y = _centor_y;
    width = _width;
    height = _height;
    x1 = centor_x-width/2;
    x2 = centor_x+width/2;
    y1 = centor_y-height/2;
    y2 = centor_y+height/2;
  }

  void update(cv::Rect2d &newRoi)
  {
    Roi = newRoi;
    update(Roi.x+Roi.width/2,Roi.y+Roi.height/2,Roi.width,Roi.height);
  }
  cv::Rect2d win2Rect()
  {
    return Roi;
  }

  int getBottomX()
  {
    return (int)centor_x;
  }
  int getBottomY()
  {
    return (int)(centor_y);
  }

};

static float IOU(window a, window b)
{
  if( a.x1 > b.x1 && a.y1 > b.y1)
  {
    window temp;
    temp = a;
    a = b;
    b = temp;
  }
  if(a.x2 < b.x1 || a.y2 < b.y1)return 0;
  else
  {
    float dx = std::min(a.x2,b.x2) - std::max(a.x1,b.x1);
    float dy = std::min(a.y2,b.y2) - std::max(a.y1,b.y1);

    return dx*dy/(a.width*a.height+b.width*b.height-dx*dy);
  }
}

#endif //UNTITLED_WINDOW_H


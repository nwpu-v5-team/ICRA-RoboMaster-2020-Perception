//
// Created by wang_shuai on 2020/8/17.
//

#include "detectionSSD.h"
#include <ros/ros.h>


//!
//! \brief  the main object detection node
//!
int main(int argc, char** argv)
{

    ros::init(argc, argv, "");

    ICRA_Vision::SSDParam   params;
    params.read("../config/config.yaml");
    ICRA_Vision::detectionSSD  worker(params);
    worker.build();

    while(1)
    {
        worker.run();
    }
}

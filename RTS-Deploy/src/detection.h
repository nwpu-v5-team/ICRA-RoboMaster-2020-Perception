//
// Created by wang_shuai on 2020/8/17.
//

#ifndef ICRA_VISION_DETECTION_H
#define ICRA_VISION_DETECTION_H


#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include "target.h"
#include <opencv2/tracking.hpp>


namespace ICRA_Vision {
    class targetBase;
    class detectionParam{

    };
    //! \brief the detectionBase with ros wrapper
    //! todo to implenment nodelet version

    class detectionBase
    {
    protected:
        std::mutex image_mutex;
        cv::Mat image;
        //! \brief NodeHandle
        ros::NodeHandle _nh;
        image_transport::ImageTransport _it;
        //! \brief tagrgetBase vector detected by detector
        std::vector<targetBase> TargetV;

    private:
        image_transport::Subscriber image_suber;


    public:
    //! \brief update the raw Image

        detectionBase() : _nh(), _it(_nh)
        {
           image_suber = _it.subscribe("", 1, &detectionBase::updateImage, this);
        }
        std::vector<targetBase>& get_TargetV()
        {
            return this->TargetV;
        }
        virtual void run() = 0;
    private:
        void updateImage(const sensor_msgs::ImageConstPtr &ptr) {
            std::lock_guard<std::mutex> guard(image_mutex);
            cv_bridge::toCvShare(ptr, ptr->encoding)->image.copyTo(image);
        }


    };






class detectionDevice : public detectionBase{
    protected:
        cv::Mat depth;
        //! \brief the subscriber for image
        image_transport::Subscriber depth_suber;
        //! \brief mutex to avoid mistake
        std::mutex depth_mutex;

    public:
        //! \brief construct function
        detectionDevice()
        {
            depth_suber = _it.subscribe("", 1, &detectionDevice::updateDepth, this);
            //image_suber = _it.subscribe("", 1, &detectionDevice::updateImage, this);
        }

        //! \brief set image , this function is used for test the fundamental usage
        void setInput(cv::Mat &src) {
            this->image = src;
        }
        //! \brief different detecter with different input process methods
        virtual void processInput() = 0;
        //! \brief run the control flow

        //! \brief update the depth Image
    private:
        void updateDepth(const sensor_msgs::ImageConstPtr &ptr) {
            std::lock_guard<std::mutex> guard(depth_mutex);
            cv_bridge::toCvShare(ptr, ptr->encoding)->image.copyTo(depth);
        }
        //! \brief convert the target vector to publish massage
        void advertise(){};

    };
}

#endif //ICRA_VISION_DETECTION_H

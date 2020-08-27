//
// Created by wang_shuai on 2020/8/17.
//

#ifndef ICRA_VISION_TARGET_H
#define ICRA_VISION_TARGET_H


namespace ICRA_Vision {
    class targetBase {
    public:
        std::string name;
        int index;
        float score;
        cv::Rect2f box;
        cv::Point2f positionP;
        cv::Point3f positionW;

        cv::Mat ROI;

        void update(const cv::Mat &Roi)
        {
            this->ROI = Roi.clone();
        }
    };

    class targetTracked : public targetBase {
        std::vector<cv::Point3f> tarjectory;
    };


    class Armor : targetTracked {

    };
}

#endif //ICRA_VISION_TARGET_H

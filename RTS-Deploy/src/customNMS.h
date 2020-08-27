//
// Created by wang_shuai on 2020/8/17.
//

#ifndef ICRA_VISION_CUSTOMNMS_H
#define ICRA_VISION_CUSTOMNMS_H

#include <driver_types.h>

template<typename ONNXType>
struct ONNXBbox
{
    ONNXType xmin,ymin,xmax,ymax;
    ONNXBbox(ONNXType xmin,
             ONNXType ymin,
             ONNXType xmax,
             ONNXType ymax)
            :
            xmin(xmin),
            xmax(xmax),
            ymin(ymin),
            ymax(ymax)
    {

    }

    ONNXBbox() = default;
};

void nms(
        cudaStream_t stream,
        const float predThre,
        const float nmsThre,
        void* box_data,
        void* score_data,
        int* index_data,
        bool* kept
);


#endif //ICRA_VISION_CUSTOMNMS_H

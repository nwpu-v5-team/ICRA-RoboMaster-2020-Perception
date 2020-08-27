//
// Created by wang_shuai on 2020/8/17.
//

#include "customNMS.h"
#include <NvInferRuntimeCommon.h>
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>
#include <array>
#include <cuda_fp16.h>

template<typename DataType>
__device__ DataType bbox_size(
        const ONNXBbox<DataType>& bbox
)
{
    return (bbox.xmax - bbox.xmin)*(bbox.ymax - bbox.ymin);
}


template<typename DataType>
__device__ DataType IOU(const ONNXBbox<DataType>&A, ONNXBbox<DataType>&B)
{
    DataType w,h;
    DataType A_area, B_area;
    DataType xmin,ymin,xmax,ymax;
    xmin = max(A.xmin, B.xmin);
    ymin = max(A.ymin, B.ymin);
    xmax = min(A.xmax, B.xmax);
    ymax = min(A.ymax, B.ymax);

    if(xmin > xmax || ymin > ymax) return 0;

    w = xmax - xmin;
    h = ymax - ymin;
    A_area =  bbox_size<DataType>(A);
    B_area =  bbox_size<DataType>(B);

    return w*h/(A_area + B_area - w*h);
}

template<typename T_BOX,typename T_SCORE,int DIM, int TSIZE>
__global__  void nmskernel(
        float nmsThres,
        float predThre,
        T_BOX* boxes_data,
        T_SCORE* scores_data,
        int* index,

        bool* afterNMS
)
{
    __shared__ bool kept[DIM*TSIZE];
    ONNXBbox<T_BOX> localBoxes[TSIZE];
    ONNXBbox<T_BOX> globalBox;
    T_SCORE  localSores[TSIZE];
    int      localIndex[TSIZE];
    T_SCORE  globalScore;
    int      globalIndex;
    // load
    const int current_idx = threadIdx.x; // + threadIdx.z*blockDim.y*blockDim.x + threadIdx.y*blockDim.x;
#pragma unroll
    for(int i=0; i<TSIZE; i++)
    {
        const int ref_idx = current_idx*TSIZE + i;
        localBoxes[i].xmin = (boxes_data)[ref_idx*4 + 0];
        localBoxes[i].ymin = (boxes_data)[ref_idx*4 + 1];
        localBoxes[i].xmax = (boxes_data)[ref_idx*4 + 2];
        localBoxes[i].ymax = (boxes_data)[ref_idx*4 + 3];
        localSores[i] = (scores_data)[ref_idx];
        localIndex[i]  = index[ref_idx];
        kept[current_idx*TSIZE + i] = (localSores[i] > predThre) && (localIndex[i] != 0);
    }

    __syncthreads();

    // compute
    for(int g=0; g<DIM*TSIZE; g++)
    {

        if(!kept[g])continue;
        globalBox.xmin = (boxes_data)[g*4 + 0];
        globalBox.ymin = (boxes_data)[g*4 + 1];
        globalBox.xmax = (boxes_data)[g*4 + 2];
        globalBox.ymax = (boxes_data)[g*4 + 3];
        globalIndex = index[g];
        globalScore = ((T_SCORE*)scores_data)[g];
        for(int i=0;i<TSIZE;i++)
        {
            if(kept[current_idx* TSIZE + i] && globalIndex == localIndex[i] && IOU(globalBox, localBoxes[i] )> nmsThres)
            {
                if(globalScore > localSores[i])
                {
                    kept[current_idx*TSIZE + i] = false;
                }
            }
        }
    }
    __syncthreads();
    // store
#pragma unroll
    for(int i=0;i<TSIZE;i++)
    {
        afterNMS[current_idx*TSIZE + i] = kept[current_idx*TSIZE+i];
    }
    __syncthreads();

}


template<typename T_BOXES, typename T_SCORES, int num>
void nmsLaunch(
        cudaStream_t stream,
        const float predThre,
        const float nmsThre,
        void* bboxes,
        void* scores,
        int* indexs,
        bool* kept)
{


#define NMS(tsize) nmskernel<T_BOXES, T_SCORES,num,(tsize)>

    void (*kernel[1])(float,float, T_BOXES*, T_SCORES*, int*, bool*) = {NMS(23)};

    kernel[0]<<<1,num,0,stream>>>(nmsThre, predThre, (T_BOXES*)bboxes, (T_SCORES*)scores, indexs, kept);
}

typedef void (*nmsFun)(cudaStream_t, float, float, void*, void*, int*, bool*);



struct nmsLaunchConfig
{
    nvinfer1::DataType T_boxes;
    nvinfer1::DataType T_scores;
    nmsFun function;

    nmsLaunchConfig(
            nvinfer1::DataType T_box,
            nvinfer1::DataType T_score,
            nmsFun fun
    )
    {
        this->T_boxes = T_box;
        this->T_scores = T_score;
        this->function = fun;
    };
    bool operator==(const nmsLaunchConfig& other)
    {
        return false;
    }
};

#define FLOAT32 nvinfer1::DataType::kFLOAT

static std::array<nmsLaunchConfig, 1> nmsLCOptions =
        {
                nmsLaunchConfig(FLOAT32,FLOAT32,nmsLaunch<float,float,178> )
        };

void nms(
        cudaStream_t stream,
        const float predThre,
        const float nmsThre,
        void* box_data,
        void* score_data,
        int* index_data,
        bool* kept
)
{
    return nmsLCOptions[0].function(
            stream,
            predThre,
            nmsThre,
            box_data,
            score_data,
            index_data,
            kept
    );
}
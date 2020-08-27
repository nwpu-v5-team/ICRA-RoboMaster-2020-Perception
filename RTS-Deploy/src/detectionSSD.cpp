//
// Created by wang_shuai on 2020/8/17.
//

#include "detectionSSD.h"
#include <cuda_runtime.h>


#include <memory>


using namespace ICRA_Vision;

/**  \brief
 **  \param  SSDParam&
 * */

detectionSSD::detectionSSD(SSDParam& params)
{
    this->mParams = params;
    host_kept = (bool*)malloc(sizeof(bool)*mParams.nmsNum);

    host_index_data = (int*)malloc(sizeof(int)*mParams.nmsNum);

    host_pred_data = (float*)malloc(sizeof(float)*mParams.nmsNum);

    host_bbox_data = (float*)malloc(sizeof(float)*mParams.nmsNum*4);

    cudaStreamCreate(&nmsStream);

    cudaMalloc(&device_kept, sizeof(bool)*mParams.nmsNum);
}

/**     \param  None
 **     \brief  to build the tensorrt engine
 **
 * */

bool detectionSSD::build()
{
    auto builder = Unique_ptr<IBuilder>(createInferBuilder(VLOG::gLogger.getTRTLogger()));

    auto expBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

    auto network = Unique_ptr<INetworkDefinition>(builder->createNetworkV2(expBatch));

    auto config = Unique_ptr<IBuilderConfig>(builder->createBuilderConfig());

    auto parser = Unique_ptr<IParser>(createParser(*network, VLOG::gLogger.getTRTLogger()));

    constructNetWork(builder, network, config, parser);
}

bool detectionSSD::teardown()
{

}
// process input cv-mat(bgr-8) to nvinfer-tensor(rgb-fp(32,16))
void detectionSSD::processInput()
{
    cv::Mat copy;
    {
        std::lock_guard<std::mutex> guard(image_mutex);
        copy = image.clone();
        copy.convertTo(copy, CV_32FC3);
    }
    //!  \brief convert data type [8uc3] -> [32fc3]


    const int inputC = mInputDims.d[1];

    const int inputH = mInputDims.d[2];

    const int inputW = mInputDims.d[3];

    const int batchSize = mParams.batchSize;// must be 1

    float* hostDataBuffer = static_cast<float*>(mBufferManager->getHostBuffer(mParams.inputTensorName));

    const int volImg = inputW*inputH*inputC;

    const int volChl = inputH*inputW;
    //! \brief opencv  h,w,c ==> tensorrt b,c,h,w
    for(int b = 0; b<batchSize; b++)
    {
        for(int h = 0; h <inputH; h++)
        {
            for(int w=0; w<inputW; w++)
            {
                cv::Vec3f* ptr = copy.ptr<cv::Vec3f>(h);

                hostDataBuffer[b*volImg + h*inputW+w] = (*(ptr +w))[2] - 123;

                hostDataBuffer[b*volImg + volChl + h*inputW+w] =  (*(ptr +w))[1] - 117;

                hostDataBuffer[b*volImg + (volChl<<1) + h*inputW+w] = (*(ptr +w))[0] - 104;// normalize
            }
        }
    }
}

//! TODO
//! TODO  need to add extra layer setting
//! TODO
void detectionSSD::constructNetWork(
        Unique_ptr<IBuilder> &builder,
        Unique_ptr<INetworkDefinition> &network,
        Unique_ptr<IBuilderConfig> &config,
        Unique_ptr<IParser> &parser)
{
    parser->parseFromFile(
            this->mParams.onnxFileName.c_str(),
            static_cast<int>(VLOG::gLogger.getReportableSeverity()));

    builder->setMaxBatchSize(mParams.batchSize);

    config->setMaxWorkspaceSize( (long long) 2<<30); // 2GB

    if(mParams.usingFP16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    if(mParams.usingDLA)
    {
        config->setDefaultDeviceType(DeviceType::kDLA);

        config->setDLACore(mParams.usingDLA);

        config->setFlag(BuilderFlag::kSTRICT_TYPES);
    }

    mEngine = std::shared_ptr<ICudaEngine>(
            builder->buildEngineWithConfig(*network, *config),
            InferDeleter()
    );
    static BufferManager bufferManager_(mEngine, mParams.batchSize);

    mBufferManager = (&bufferManager_);
    return;
}
//!
//! \brief copy device memory to host memory
//!

void detectionSSD::copyDevice2Host()
{
    cudaMemcpyAsync(host_bbox_data, device_bbox_data, sizeof(float)*mParams.nmsNum*4, cudaMemcpyDeviceToHost, nmsStream);

    cudaMemcpyAsync(host_pred_data, device_pred_data, sizeof(float)*mParams.nmsNum, cudaMemcpyDeviceToHost, nmsStream);

    cudaMemcpyAsync(host_index_data, device_index_data, sizeof(int)*mParams.nmsNum, cudaMemcpyDeviceToHost, nmsStream);

    cudaMemcpyAsync(host_kept, device_kept, sizeof(bool)*mParams.nmsNum, cudaMemcpyDeviceToHost, nmsStream);
}

void detectionSSD::processOutput()
{
    TargetV.clear();
    for(int i=0; i<mParams.nmsNum; i++)
    {
        if(host_kept[i] && host_index_data[i])
        {
            targetBase tb;

            tb.score = host_pred_data[i];

            tb.index = host_index_data[i];

            tb.name = mParams.Index2class[tb.index];

            tb.box.x = host_bbox_data[4*i + 0] * image.cols;

            tb.box.y = host_bbox_data[4*i + 1] * image.rows;

            tb.box.width = (host_bbox_data[4*i + 2]* image.cols - tb.box.x);

            tb.box.height = (host_bbox_data[4*i + 3]* image.rows - tb.box.y);

            tb.positionP.x = (host_bbox_data[4*i] + host_bbox_data[4*i+2])/2;

            tb.positionP.y = (host_bbox_data[4*i +1] + host_bbox_data[4*i+3])/2;

            TargetV.push_back(tb);

        }
    }
    cv::Mat test = image.clone();

    cv::putText(test, "time:"+std::to_string(timeCost)+"ms",cv::Point(20,20),1,1,cv::Scalar(255,255,255));
    for(int i=0;i<TargetV.size(); i++)
    {
        cv::rectangle(test, TargetV[i].box, cv::Scalar(255,255),1);

        cv::putText(test, TargetV[i].name, TargetV[i].box.tl(), 1,0.5,cv::Scalar(200,100,100));
    }
    cv::imshow("test", test);

    cv::waitKey(1);
}

void detectionSSD::detect()
{

    auto context = Unique_ptr<IExecutionContext>(mEngine->createExecutionContext());

    mBufferManager->copyInputToDevice();

    clock_t  start, end;

    start = clock();

    bool status = context->execute(mParams.batchSize, mBufferManager->getDeviceBindings().data());

    device_pred_data = (mBufferManager->getDeviceBuffer(mParams.outputTensorName[1]));

    device_bbox_data = (mBufferManager->getDeviceBuffer(mParams.outputTensorName[0]));

    device_index_data = static_cast<int*>(mBufferManager->getDeviceBuffer(mParams.outputTensorName[2]));

    nms(nmsStream, mParams.conf_threshold, mParams.nms_threshold, device_bbox_data, device_pred_data, device_index_data,device_kept);

    end = clock();

    timeCost = double(end - start)/CLOCKS_PER_SEC*1000;

    copyDevice2Host();
}


void detectionSSD::load()
{

}

void detectionSSD::store()
{
    auto hostMemory = mEngine->serialize();
}
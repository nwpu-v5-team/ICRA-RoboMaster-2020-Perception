//
// Created by wang_shuai on 2020/8/17.
//

#ifndef ICRA_VISION_DETECTIONSSD_H
#define ICRA_VISION_DETECTIONSSD_H

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include "detection.h"
#include <iostream>
#include <cstdlib>
#include <iostream>

#include "bufferManager.h"
#include "target.h"
#include "logger.h"
#include "customNMS.h"
#include <cuda_runtime_api.h>
#include <yaml-cpp/yaml.h>

using namespace nvonnxparser;
using namespace nvinfer1;



namespace ICRA_Vision {

    class targetBase;
    //! \brief the param class for SSD-detector
    class SSDParam : detectionParam {
    public:
        //! \brief cache the engine to avoid the re-build of engine
        std::string cacheFile;
        //! \brief to locate the onnx file
        std::string onnxFileName;
        //! \brief the name of inputTensor [input.1]
        std::string inputTensorName;
        //! \brief the name of outputTensor [bounding box , scores, class]
        std::vector<std::string> outputTensorName;
        //! \brief keep top k items of the nms result
        int keepTopK;
        //! \brief the batchSize of the network (for inference must be 1 !!!)
        int batchSize;
        //! \brief the threshold of no-max-suppress
        float nms_threshold;
        //! \brief the threshold of confidence of classification
        float conf_threshold;
        //! \brief datatype of the Network input
        DataType dataType;
        //! \brief whether to use DLA to accerlate
        int usingDLA;
        //! \brief use FP16 to infer or not
        bool usingFP16;
        //! \brief the total num of no-max-suppress task
        int nmsNum = 4094;
        //! \brief map the class index to class string
        std::map<int, std::string> Index2class;

    public:
        //! \brief read the yaml config file to build params
        void read(std::string filePath) {
            YAML::Node readFile = YAML::LoadFile(filePath);
            //cacheFile     = readFile["cacheFile"].as<std::string>();

            onnxFileName = readFile["onnxFileName"].as<std::string>();

            keepTopK = readFile["keepTopK"].as<int>();

            batchSize = readFile["batchSize"].as<int>();

            nms_threshold = readFile["nms_threshold"].as<float>();

            conf_threshold = readFile["conf_threshold"].as<float>();

            usingDLA = readFile["usingDLA"].as<int>();

            usingFP16 = readFile["usingFP16"].as<bool>();

            std::vector<std::string> classCat;

            classCat = readFile["class"].as<std::vector<std::string>>();

            inputTensorName = readFile["input"].as<std::string>();

            outputTensorName = readFile["output"].as<std::vector<std::string>>();


            for (int i = 0; i < classCat.size(); i++) {

                Index2class[i] = classCat[i];

            }
        }
        //!  \brief to display the params of SSD-detector
        friend std::ostream &operator<<(std::ostream &os, SSDParam &param) {
            os  <<  "-------------+param-setting+----------" << std::endl;

            os  <<   "onnxFileName : " << param.onnxFileName << std::endl;

            os  <<   "keepTopK : " << param.keepTopK         << std::endl;

            os  << "batchSize : " << param.batchSize         << std::endl;

            os  << "nms_threshold : " << param.nms_threshold << std::endl;

            os  << "conf_threshold : " << param.conf_threshold << std::endl;

            os  << "usingFP16 : " << param.usingFP16         << std::endl;
            return os;
        }
    };

    //! \brief unique ptr construct helper
    struct InferDeleter {
        template<typename T>
        void operator()(T *obj) const {
            if (obj) {
                obj->destroy();
            }
        }
    };

    //! \brief detectionSSD methods implements
    class detectionSSD : public detectionDevice {
        template<typename T>
        using Unique_ptr = std::unique_ptr<T, InferDeleter>;
    public:

        explicit detectionSSD(SSDParam &param);
        //! \brief build up the engine
        bool build();
        //! \brief process cv::Mat to nvinfer::Tensor
        void processInput() override;
        //! \brief construct and configure the network
        void constructNetWork(
                Unique_ptr<IBuilder> &builder,
                Unique_ptr<INetworkDefinition> &network,
                Unique_ptr<IBuilderConfig> &config,
                Unique_ptr<IParser> &parser
        );
        //! \brief detect work flow
        void detect();

        //! \brief control flow of SSD-detector method
        void run() override {
            processInput();
            detect();
            processOutput();
        }
        //! \brief shut up the fucking tensorrt engine
        bool teardown();
        //! \brief free the memory on device and host
        ~detectionSSD() {
            cudaFree(device_bbox_data);
            cudaFree(device_index_data);
            cudaFree(device_kept);
            cudaFree(device_pred_data);

            free(host_bbox_data);
            free(host_pred_data);
            free(host_index_data);
            free(host_kept);

            cudaStreamDestroy(nmsStream);
        }

    private:
        //! \brief ssd engine
        std::shared_ptr<ICudaEngine> mEngine;
        //! \brief ssd param
        SSDParam mParams;
        //! \brief buffer manager
        BufferManager *mBufferManager;
        //! \brief input dimension
        nvinfer1::Dims mInputDims = {4, 1, 3, 512, 512};
        //! ptr
        //! \details
        //! pred, bbox, index,
        float *host_pred_data;
        void *device_pred_data;
        float *host_bbox_data;
        void *device_bbox_data;
        int *host_index_data, *device_index_data;
        bool *host_kept, *device_kept;
        //! \brief cudaStream for nms task
        cudaStream_t nmsStream;
        //! \brief record inference time
        double timeCost;
        //! \brief transport memory from device to Host
        void copyDevice2Host();

    public:
        //! \brief process output
        void processOutput();
        //! \brief display
        friend std::ostream &operator<<(std::ostream &os, detectionSSD &ssd) {
            os << ssd.mParams << std::endl;
        }
        //! \brief load cache file
        void load();
        //! \brief store cache file
        void store();

    };

}

#endif //ICRA_VISION_DETECTIONSSD_H

//
// Created by wang_shuai on 2020/8/17.
//

#ifndef ICRA_VISION_LOGGER_H
#define ICRA_VISION_LOGGER_H

#include "logging.h"
namespace VLOG
{
    extern Logger gLogger;
    extern LogStreamConsumer gLogVerbose;
    extern LogStreamConsumer gLogInfo;
    extern LogStreamConsumer gLogWarning;
    extern LogStreamConsumer gLogError;
    extern LogStreamConsumer gLogFatal;

    void setReportableSeverity(Logger::Severity severity);
}

#endif //ICRA_VISION_LOGGER_H

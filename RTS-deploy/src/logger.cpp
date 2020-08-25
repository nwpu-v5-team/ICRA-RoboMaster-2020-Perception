//
// Created by wang_shuai on 2020/8/17.
//

#include "logger.h"

namespace VLOG
{
    Logger gLogger{Logger::Severity::kERROR};
    LogStreamConsumer gLogVerbose{LOG_VERBOSE(gLogger)};
    LogStreamConsumer gLogInfo{LOG_INFO(gLogger)};
    LogStreamConsumer gLogWarning{LOG_WARN(gLogger)};
    LogStreamConsumer gLogError{LOG_ERROR(gLogger)};
    LogStreamConsumer gLogFatal{LOG_FATAL(gLogger)};

    void setReportableSeverity(Logger::Severity severity)
    {
        gLogger.setReportableSeverity(severity);
        gLogVerbose.setReportableSeverity(severity);
        gLogInfo.setReportableSeverity(severity);
        gLogWarning.setReportableSeverity(severity);
        gLogError.setReportableSeverity(severity);
        gLogFatal.setReportableSeverity(severity);
    }
}
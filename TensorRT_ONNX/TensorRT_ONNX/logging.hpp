#pragma once

#include "logger.h"
#include "ErrorRecorder.h"
#include "logging.h"

SampleErrorRecorder gRecorder;
namespace sample
{
    Logger gLogger{ Logger::Severity::kVERBOSE };
    //Logger gLogger{ Logger::Severity::kINFO };
    //Logger gLogger{ Logger::Severity::kWARNING };
    //Logger gLogger{ Logger::Severity::kERROR };
    LogStreamConsumer gLogVerbose{ LOG_VERBOSE(gLogger) };
    LogStreamConsumer gLogInfo{ LOG_INFO(gLogger) };
    LogStreamConsumer gLogWarning{ LOG_WARN(gLogger) };
    LogStreamConsumer gLogError{ LOG_ERROR(gLogger) };
    LogStreamConsumer gLogFatal{ LOG_FATAL(gLogger) };

    void setReportableSeverity(Logger::Severity severity)
    {
        gLogger.setReportableSeverity(severity);
        gLogVerbose.setReportableSeverity(severity);
        gLogInfo.setReportableSeverity(severity);
        gLogWarning.setReportableSeverity(severity);
        gLogError.setReportableSeverity(severity);
        gLogFatal.setReportableSeverity(severity);
    }
} // namespace sample



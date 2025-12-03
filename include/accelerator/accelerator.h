#pragma once

#include <functional>
#include <future>
#include <map>
#include <mutex>
#include <string>
#include <vector>

enum class AcceleratorKind { CPU, DSP, GPU, NPU };

struct Workload {
    std::string name;
    AcceleratorKind preferred;
    std::function<void()> task;
};

struct AcceleratorMetrics {
    double utilization = 0.0;
    double latencyMs = 0.0;
};

struct AcceleratorConfig {
    AcceleratorKind kind;
    int maxConcurrent = 1;
    bool available = true;
};

class AcceleratorRuntime {
public:
    virtual ~AcceleratorRuntime() = default;
    virtual std::future<void> dispatch(const Workload &workload) = 0;
    virtual AcceleratorMetrics latestMetrics(AcceleratorKind kind) const = 0;
};

class QualcommGen2Runtime : public AcceleratorRuntime {
public:
    explicit QualcommGen2Runtime(std::vector<AcceleratorConfig> configs = {});

    std::future<void> dispatch(const Workload &workload) override;
    AcceleratorMetrics latestMetrics(AcceleratorKind kind) const override;

    void setPowerMode(const std::string &mode);
    void logTopology() const;

private:
    struct DeviceState {
        AcceleratorConfig config;
        mutable std::mutex mutex;
        AcceleratorMetrics metrics;
    };

    std::map<AcceleratorKind, DeviceState> devices_;
    std::future<void> launchAsync(const Workload &workload, DeviceState &device);
};

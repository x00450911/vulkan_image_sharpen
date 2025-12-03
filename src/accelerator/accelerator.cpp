#include "accelerator/accelerator.h"

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <thread>

namespace {
std::string KindToString(AcceleratorKind kind) {
    switch (kind) {
    case AcceleratorKind::CPU:
        return "CPU";
    case AcceleratorKind::DSP:
        return "DSP";
    case AcceleratorKind::GPU:
        return "GPU";
    case AcceleratorKind::NPU:
        return "NPU";
    }
    return "UNKNOWN";
}
}

QualcommGen2Runtime::QualcommGen2Runtime(std::vector<AcceleratorConfig> configs) {
    if (configs.empty()) {
        configs = {
            {AcceleratorKind::CPU, 4, true},
            {AcceleratorKind::DSP, 2, true},
            {AcceleratorKind::GPU, 2, true},
            {AcceleratorKind::NPU, 1, true},
        };
    }

    for (auto &cfg : configs) {
        devices_.emplace(cfg.kind, DeviceState{cfg});
    }
}

std::future<void> QualcommGen2Runtime::dispatch(const Workload &workload) {
    auto it = devices_.find(workload.preferred);
    if (it == devices_.end()) {
        throw std::runtime_error("Requested accelerator not available");
    }
    if (!workload.task) {
        throw std::invalid_argument("Workload has no task callback");
    }
    return launchAsync(workload, it->second);
}

AcceleratorMetrics QualcommGen2Runtime::latestMetrics(AcceleratorKind kind) const {
    auto it = devices_.find(kind);
    if (it == devices_.end()) {
        return {};
    }
    std::lock_guard<std::mutex> lock(it->second.mutex);
    return it->second.metrics;
}

void QualcommGen2Runtime::setPowerMode(const std::string &mode) {
    std::cout << "[QCOM] Switching power mode to " << mode << '\n';
}

void QualcommGen2Runtime::logTopology() const {
    std::cout << "[QCOM] Active accelerators:" << '\n';
    for (const auto &entry : devices_) {
        const auto &cfg = entry.second.config;
        std::cout << "  * " << KindToString(cfg.kind) << " (concurrency=" << cfg.maxConcurrent
                  << ", available=" << (cfg.available ? "yes" : "no") << ")\n";
    }
}

std::future<void> QualcommGen2Runtime::launchAsync(const Workload &workload, DeviceState &device) {
    return std::async(std::launch::async, [this, workload, &device]() {
        if (!device.config.available) {
            throw std::runtime_error("Accelerator offline: " + KindToString(device.config.kind));
        }
        const auto start = std::chrono::steady_clock::now();
        std::cout << "[" << KindToString(device.config.kind) << "] Running " << workload.name << '\n';
        workload.task();
        const auto end = std::chrono::steady_clock::now();
        const double elapsedMs = std::chrono::duration<double, std::milli>(end - start).count();
        {
            std::lock_guard<std::mutex> lock(device.mutex);
            device.metrics.latencyMs = elapsedMs;
            device.metrics.utilization = std::min(1.0, elapsedMs / 16.0); // pretend 16ms budget.
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        std::cout << "[" << KindToString(device.config.kind) << "] Completed " << workload.name
                  << " in " << elapsedMs << " ms" << '\n';
    });
}

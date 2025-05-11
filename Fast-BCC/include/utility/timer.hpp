#ifndef TIMER_H
#define TIMER_H

#include <chrono>
#include <unordered_map>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <iostream>

class Timer {
private:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = std::chrono::time_point<Clock>;
    using Duration = std::chrono::duration<double>;

    TimePoint start_time;

public:
    Timer() : start_time(Clock::now()) {}

    void reset() {
        start_time = Clock::now();
    }

    double stop() const {
        return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start_time).count();
    }
};

class TimerManager {
private:
    std::unordered_map<std::string, double> function_times;

public:
    void add_function_time(const std::string& function_name, double time) {
        function_times[function_name] += time;
    }

    void print_times() const {
        std::vector<std::pair<std::string, double>> sorted_times(function_times.begin(), function_times.end());

        std::sort(sorted_times.begin(), sorted_times.end(), [](const auto& a, const auto& b) {
            return a.second > b.second;  
        });

        std::cout << "\nFunction execution times:\n";
        for (const auto& pair : sorted_times) {
            std::cout << std::fixed << std::setprecision(2)
                      << "Time of " << pair.first << " : " << pair.second << " ms\n";
        }
        std::cout << std::endl;
    }

    void reset_function_times() {
        function_times.clear();
    }

    void print_total_function_time(const std::string& function_name) const {
        double total_time = 0;
        for (const auto& pair : function_times) {
            total_time += pair.second;
        }

        std::cout << "\nTotal execution time for " << function_name << ": " << total_time << " ms.\n";
        print_times();
    }
};

#endif // TIMER_H

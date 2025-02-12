#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>

void printProgress(double percentage, double transferredMiB, double totalMiB, double speedMiBs) {
    // Calculate remaining time in seconds
    double remainingMiB = totalMiB - transferredMiB;
    int remainingTime = static_cast<int>(remainingMiB / speedMiBs);

    // Calculate time components
    int minutes = remainingTime / 60;
    int seconds = remainingTime % 60;

    // Clear the current line
    std::cout << "\r";

    // Print the progress
    std::cout << std::fixed << std::setprecision(1) << percentage * 100 << "%\t"
              << transferredMiB << " MiB / " << totalMiB << " MiB = "
              << std::setprecision(3) << percentage << "   "
              << speedMiBs << " MiB/s       "
              << minutes << ":" << std::setw(2) << std::setfill('0') << seconds
              << "    ";

    // Flush the output to ensure it's displayed immediately
    std::cout << std::flush;
}

int main() {
    double totalMiB = 5740.2;
    double transferredMiB = 0.0;
    double speedMiBPerSecond = 9.6;

    // Simulate a process that updates regularly
    for (int i = 0; i <= 100; ++i) {
        double percentage = transferredMiB / totalMiB;
        printProgress(percentage, transferredMiB, totalMiB, speedMiBPerSecond);

        // Simulate doing work by sleeping
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // Update transferred amount (simulate transfer)
        transferredMiB += speedMiBPerSecond * 0.1; // Update every 0.1 seconds
    }

    // Ensure to print a newline once done
    std::cout << std::endl;

    return 0;
}

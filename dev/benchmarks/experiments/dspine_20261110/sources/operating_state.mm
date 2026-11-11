#import <Foundation/Foundation.h>
#include <iostream>
int main() {
  @autoreleasepool {
    std::cout << "{\"thermal_state\":" << static_cast<unsigned>(NSProcessInfo.processInfo.thermalState)
              << ",\"uptime_seconds\":" << NSProcessInfo.processInfo.systemUptime << "}\n";
  }
}

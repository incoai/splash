# Standalone CPU-only build: no Metal backend library or model/GPU execution.
# make -f dev/tests/engine/flash_request_command_trace_test.mk test
CXX := xcrun clang++
BUILD ?= build/flash-request-command-trace-tests
TEST := $(BUILD)/flash-request-command-trace-test
CXXFLAGS := -std=c++20 -O2 -Wall -Wextra -Werror -fobjc-arc -Iruntime
HEADERS := runtime/flash/FlashRequestCommandTrace.hpp runtime/metal/ProfilingJson.hpp \
	runtime/metal/MetalBackend.hpp runtime/engine/Json.hpp

.PHONY: all test clean
all: $(TEST)

$(TEST): dev/tests/engine/flash_request_command_trace_test.mm $(HEADERS)
	mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) $< -framework Foundation -o $@

test: $(TEST)
	$(TEST)

clean:
	rm -f $(TEST)

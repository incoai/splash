ENGINE_TEST_BUILD := $(BUILD)/engine-tests
ENGINE_TEST_CXXFLAGS := -std=c++20 -O2 -Wall -Wextra -Werror -Iruntime -Idev \
	$(MACOS_TARGET_FLAG)
# Test kernels compile like production ones without the release optimizer.
TEST_METALFLAGS := $(filter-out -O3,$(PROD_METALFLAGS))
# Include the shared tools and production flags: some tests also link the
# production library or metallib, and all share ENGINE_LINKFLAGS.
TEST_CONFIG_DIGEST := $(shell printf '%s\0' $(CONFIG_DIGEST) \
	$(call shell-quote,$(ENGINE_TEST_CXXFLAGS)) \
	$(call shell-quote,$(TEST_METALFLAGS)) | shasum -a 256 | cut -c1-16)
ENGINE_SANITIZER_BUILD := $(BUILD)/sanitizers
ENGINE_SANITIZER_CXXFLAGS := -std=c++20 -O1 -g -fno-omit-frame-pointer \
	-Wall -Wextra -Werror -Iruntime -Idev $(MACOS_TARGET_FLAG)
SANITIZER_CONFIG_DIGEST := $(shell printf '%s\0' $(CONFIG_DIGEST) \
	$(call shell-quote,$(ENGINE_SANITIZER_CXXFLAGS)) \
	| shasum -a 256 | cut -c1-16)
# Offline kernel tuning lives outside the production library. The dev tool
# and its tests compile these sources directly.
TUNING_SOURCES := \
	dev/tuning/Tuning.cpp \
	dev/tuning/Measurement.cpp \
	dev/tuning/LinearTuning.cpp \
	dev/tuning/AttentionTuning.cpp \
	dev/tuning/DraftAttentionTuning.cpp \
	dev/tuning/MoeTuning.cpp \
	dev/tuning/TuningWorkloads.cpp
BACKEND_CONTROL_SOURCES := \
	runtime/engine/Scheduler.cpp \
	runtime/model/DraftContextPlan.cpp \
	runtime/engine/KvPool.cpp \
	runtime/engine/KvCache.cpp \
	runtime/engine/StateCache.cpp \
	runtime/engine/Cache.cpp \
	runtime/engine/Engine.cpp
MODEL_SOURCES := \
	runtime/model/WeightStore.cpp \
	runtime/model/Qwen3_6Moe.cpp \
	runtime/model/Qwen3_8.cpp \
	runtime/model/QwenVision.cpp \
	runtime/model/QwenTarget.cpp \
	runtime/model/DFlashDraft.cpp \
	runtime/model/ModelFactory.cpp \
	runtime/model/ModelDescriptor.mm
MODEL_OPERATOR_SOURCES := \
	runtime/ops/DraftAttention.cpp \
	runtime/ops/Embedding.cpp \
	runtime/ops/ExecutionPlans.cpp \
	runtime/ops/GDN.cpp \
	runtime/ops/Linear.cpp \
	runtime/ops/MoE.cpp \
	runtime/ops/Normalization.cpp \
	runtime/ops/PagedAttention.cpp \
	runtime/ops/Sampling.cpp
TEST_BACKEND_ASAN := $(ENGINE_SANITIZER_BUILD)/kv-first-engine-asan-ubsan
TEST_BACKEND_TSAN := $(ENGINE_SANITIZER_BUILD)/kv-first-engine-tsan
TEST_FD_TRANSPORT_ASAN := $(ENGINE_SANITIZER_BUILD)/native-fd-asan-ubsan
TEST_FD_TRANSPORT_TSAN := $(ENGINE_SANITIZER_BUILD)/native-fd-tsan
TEST_OPERATOR_TUNING_ASAN := $(ENGINE_SANITIZER_BUILD)/operator-tuning-asan-ubsan
TEST_OPERATOR_TUNING_TSAN := $(ENGINE_SANITIZER_BUILD)/operator-tuning-tsan
TEST_OPERATOR_MEASUREMENT_ASAN := $(ENGINE_SANITIZER_BUILD)/operator-measurement-asan-ubsan
TEST_OPERATOR_MEASUREMENT_TSAN := $(ENGINE_SANITIZER_BUILD)/operator-measurement-tsan
TEST_MEMORY_TEST := $(ENGINE_TEST_BUILD)/engine-memory-plan
TEST_DEVICE_QUERIES := $(ENGINE_TEST_BUILD)/device-queries
TEST_KV_PAGE_CACHE_TEST := $(ENGINE_TEST_BUILD)/kv-page-cache
TEST_KV_FIRST_CACHE_TEST := $(ENGINE_TEST_BUILD)/kv-first-cache
TEST_DRAFT_CONTEXT_PLAN_TEST := $(ENGINE_TEST_BUILD)/draft-context-plan
TEST_RAGGED_SCHEDULER_TEST := $(ENGINE_TEST_BUILD)/ragged-scheduler
TEST_CACHE_TEST := $(ENGINE_TEST_BUILD)/engine-cache
TEST_KV_FIRST_ENGINE_TEST := $(ENGINE_TEST_BUILD)/kv-first-engine
TEST_ELASTIC_KV_TEST := $(ENGINE_TEST_BUILD)/elastic-kv-pool
TEST_PROTOCOL_TEST := $(ENGINE_TEST_BUILD)/protocol
TEST_NATIVE_LOOP_TEST := $(ENGINE_TEST_BUILD)/native-engine-loop
TEST_FD_TRANSPORT_TEST := $(ENGINE_TEST_BUILD)/native-fd-transport
TEST_BOOTSTRAP_TEST := $(ENGINE_TEST_BUILD)/runtime-bootstrap
TEST_RESOURCES_TEST := $(ENGINE_TEST_BUILD)/runtime-resources
TEST_METRICS_TEST := $(ENGINE_TEST_BUILD)/runtime-metrics
TEST_MODEL_PACKAGE_TEST := $(ENGINE_TEST_BUILD)/model-package
TEST_MEMORY_AUDIT_TEST := $(ENGINE_TEST_BUILD)/memory-audit
TEST_QWEN_STATE_TEST := $(ENGINE_TEST_BUILD)/qwen-state-storage
TEST_STATUS_TEST := $(ENGINE_TEST_BUILD)/runtime-status
TEST_Q8_CPU_TEST := $(ENGINE_TEST_BUILD)/q8-paged-kv
TEST_Q8_METAL_TEST := $(ENGINE_TEST_BUILD)/q8-paged-kv-metal
TEST_Q8_STORAGE_TEST := $(ENGINE_TEST_BUILD)/q8-page-storage
TEST_Q8_ATTENTION_TEST := $(ENGINE_TEST_BUILD)/q8-flash-attention
TEST_Q8_PREFILL_TEST := $(ENGINE_TEST_BUILD)/q8-chunked-prefill
TEST_Q4_SGMATRIX_TEST := $(ENGINE_TEST_BUILD)/q4-sgmatrix
TEST_Q4_BATCH_TEST := $(ENGINE_TEST_BUILD)/q4-batched-projection
TEST_Q4_PREFILL_TEST := $(ENGINE_TEST_BUILD)/q4-prefill-projection
TEST_MOE_METAL_TEST := $(ENGINE_TEST_BUILD)/moe-metal
TEST_GDN_METAL_TEST := $(ENGINE_TEST_BUILD)/gdn-metal
TEST_OPERATOR_WORKSPACE := $(ENGINE_TEST_BUILD)/operator-workspace
TEST_OPERATOR_TUNING := $(ENGINE_TEST_BUILD)/operator-tuning
TEST_OPERATOR_MEASUREMENT := $(ENGINE_TEST_BUILD)/operator-measurement
TEST_EXECUTION_PLANS := $(ENGINE_TEST_BUILD)/execution-plans
TEST_MODEL_EXECUTION_PLANS := $(ENGINE_TEST_BUILD)/model-execution-plans
TEST_ATTENTION_PLAN := $(ENGINE_TEST_BUILD)/paged-attention-plan
TEST_LINEAR_PLAN := $(ENGINE_TEST_BUILD)/linear-plan
TEST_LINEAR_TUNING := $(ENGINE_TEST_BUILD)/linear-tuning
TEST_ATTENTION_TUNING := $(ENGINE_TEST_BUILD)/attention-tuning
TEST_DRAFT_ATTENTION_TUNING := $(ENGINE_TEST_BUILD)/draft-attention-tuning
TEST_MOE_TUNING := $(ENGINE_TEST_BUILD)/moe-tuning
TEST_TUNING_WORKLOADS := $(ENGINE_TEST_BUILD)/tuning-workloads
TUNE_KERNELS := $(ENGINE_TEST_BUILD)/tune-kernels
TEST_DFLASH_BATCH_CONTROL_TEST := $(ENGINE_TEST_BUILD)/dflash-batch-control
TEST_DRAFT_ATTENTION_TEST := $(ENGINE_TEST_BUILD)/draft-attention
TEST_GDN_DECODE_TEST := $(ENGINE_TEST_BUILD)/gdn-decode
TEST_DRAFT_SELECTOR_TEST := $(ENGINE_TEST_BUILD)/draft-selector
TEST_Q4_PREFILL_PROFILE := $(ENGINE_TEST_BUILD)/q4-prefill-profile
TEST_Q4_DECODE_PROFILE := $(ENGINE_TEST_BUILD)/q4-decode-profile
TEST_BACKEND_BENCHMARK := $(ENGINE_TEST_BUILD)/backend-benchmark
TEST_DECODE_PROFILE := $(ENGINE_TEST_BUILD)/decode-profile
TEST_ATTENTION_SWEEP := $(ENGINE_TEST_BUILD)/attention-sweep
TEST_MODEL_RUNTIME_ORACLE := $(ENGINE_TEST_BUILD)/model-runtime-oracle
TEST_VISION_ENCODER_TEST := $(ENGINE_TEST_BUILD)/vision-encoder
TEST_Q8_AIR := $(ENGINE_TEST_BUILD)/q8-paged-kv.air
TEST_Q8_LIB := $(ENGINE_TEST_BUILD)/q8-paged-kv.metallib
# Both Q8 tests share the attention and store kernels of both phases.
TEST_Q8_KERNEL_SOURCES := \
	runtime/metal/kernels/prefill/attention_q8.metal \
	runtime/metal/kernels/decode/attention_q8.metal \
	runtime/metal/kernels/prefill/attention_q8_store.metal \
	runtime/metal/kernels/decode/attention_q8_store.metal
TEST_Q8_KERNEL_AIRS := $(patsubst runtime/metal/kernels/%.metal,$(ENGINE_TEST_BUILD)/kernels/%.air,$(TEST_Q8_KERNEL_SOURCES))
TEST_Q8_ATTENTION_LIB := $(ENGINE_TEST_BUILD)/q8-attention.metallib
TEST_METAL_BACKEND_TEST := $(ENGINE_TEST_BUILD)/metal-backend
TEST_METAL_BACKEND_AIR := $(ENGINE_TEST_BUILD)/metal-backend.air
TEST_METAL_BACKEND_LIB := $(ENGINE_TEST_BUILD)/metal-backend.metallib

TEST_CPU_TARGETS := $(TEST_OPERATOR_WORKSPACE) \
	$(TEST_DEVICE_QUERIES) \
	$(TEST_TUNING_WORKLOADS) \
	$(TEST_LINEAR_PLAN) $(TEST_LINEAR_TUNING) $(TEST_ATTENTION_TUNING) \
	$(TEST_DRAFT_ATTENTION_TUNING) $(TEST_MOE_TUNING) \
	$(TEST_OPERATOR_TUNING) \
	$(TEST_OPERATOR_MEASUREMENT) \
	$(TEST_EXECUTION_PLANS) \
	$(TEST_MODEL_EXECUTION_PLANS) \
	$(TEST_MEMORY_TEST) \
	$(TEST_KV_PAGE_CACHE_TEST) \
	$(TEST_KV_FIRST_CACHE_TEST) \
	$(TEST_DRAFT_CONTEXT_PLAN_TEST) \
	$(TEST_RAGGED_SCHEDULER_TEST) \
	$(TEST_CACHE_TEST) \
	$(TEST_KV_FIRST_ENGINE_TEST) \
	$(TEST_ELASTIC_KV_TEST) \
	$(TEST_PROTOCOL_TEST) \
	$(TEST_NATIVE_LOOP_TEST) \
	$(TEST_FD_TRANSPORT_TEST) \
	$(TEST_BOOTSTRAP_TEST) \
	$(TEST_METRICS_TEST) \
	$(TEST_MEMORY_AUDIT_TEST) \
	$(TEST_STATUS_TEST) \
	$(TEST_Q8_CPU_TEST)

TEST_METAL_TARGETS := $(TEST_Q4_SGMATRIX_TEST) $(TEST_TUNING_WORKLOADS) \
	$(TEST_LINEAR_TUNING) \
	$(TEST_ATTENTION_TUNING) \
	$(TEST_DRAFT_ATTENTION_TUNING) \
	$(TEST_MOE_TUNING) \
	$(TEST_ATTENTION_PLAN) \
	$(TEST_LINEAR_PLAN) \
	$(TEST_RESOURCES_TEST) \
	$(TEST_MODEL_PACKAGE_TEST) \
	$(TEST_QWEN_STATE_TEST) \
	$(TEST_Q8_METAL_TEST) \
	$(TEST_Q8_STORAGE_TEST) \
	$(TEST_Q8_ATTENTION_TEST) \
	$(TEST_Q8_PREFILL_TEST) \
	$(TEST_Q4_BATCH_TEST) \
	$(TEST_Q4_PREFILL_TEST) \
	$(TEST_MOE_METAL_TEST) \
	$(TEST_GDN_METAL_TEST) \
	$(TEST_DFLASH_BATCH_CONTROL_TEST) \
	$(TEST_DRAFT_ATTENTION_TEST) \
	$(TEST_GDN_DECODE_TEST) \
	$(TEST_DRAFT_SELECTOR_TEST) \
	$(TEST_METAL_BACKEND_TEST) \
	$(LIB) $(TEST_METAL_BACKEND_LIB) $(TEST_Q8_LIB) $(TEST_Q8_ATTENTION_LIB)

TEST_UNIT_TEST_TARGETS := $(sort $(TEST_CPU_TARGETS) $(TEST_METAL_TARGETS))

# Keep every output that uses a flag set together, including standalone
# benchmarks, real-model tests and intermediate test AIRs/metallibs.
TEST_CONFIG_TARGETS := $(filter-out $(LIB),$(TEST_UNIT_TEST_TARGETS)) \
	$(TEST_MODEL_RUNTIME_ORACLE) $(TEST_VISION_ENCODER_TEST) \
	$(TEST_DECODE_PROFILE) $(TEST_ATTENTION_SWEEP) \
	$(TEST_Q8_AIR) $(TEST_Q8_KERNEL_AIRS) $(TEST_METAL_BACKEND_AIR)
PRODUCTION_CONFIG_TARGETS += $(TEST_Q4_PREFILL_PROFILE) \
	$(TEST_Q4_DECODE_PROFILE) $(TEST_BACKEND_BENCHMARK) $(TUNE_KERNELS)
SANITIZER_CONFIG_TARGETS := $(TEST_BACKEND_ASAN) $(TEST_BACKEND_TSAN) \
	$(TEST_FD_TRANSPORT_ASAN) $(TEST_FD_TRANSPORT_TSAN) \
	$(TEST_OPERATOR_TUNING_ASAN) $(TEST_OPERATOR_TUNING_TSAN) \
	$(TEST_OPERATOR_MEASUREMENT_ASAN) $(TEST_OPERATOR_MEASUREMENT_TSAN)

# These binaries compile runtime, tuning and test sources directly. Header
# prerequisites and the force dependency are not compiler input files.
ENGINE_TEST_HEADERS := $(filter %.h %.hpp,$(PRODUCTION_ENGINE_INPUTS)) \
	$(wildcard dev/tuning/*.hpp dev/tests/engine/*.hpp)
TEST_INPUTS = $(filter-out %.h %.hpp %.metallib,$(BUILD_INPUTS))
$(filter-out %.air %.metallib,$(TEST_CONFIG_TARGETS)) \
	$(TEST_Q4_PREFILL_PROFILE) $(TEST_Q4_DECODE_PROFILE) \
	$(TEST_BACKEND_BENCHMARK) $(TUNE_KERNELS) $(SANITIZER_CONFIG_TARGETS): \
	$(ENGINE_TEST_HEADERS)

$(ENGINE_TEST_BUILD):
	mkdir -p $@

$(ENGINE_SANITIZER_BUILD):
	mkdir -p $@

$(TEST_DEVICE_QUERIES): dev/tests/engine/device_queries_test.mm | $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_TEST_CXXFLAGS) -fobjc-arc $(TEST_INPUTS) \
		-framework Foundation -o $@

$(TEST_MEMORY_TEST): runtime/metal/DeviceCapabilities.cpp \
		runtime/engine/MemoryPlan.cpp \
		dev/tests/engine/engine_memory_plan_test.cpp | $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_TEST_CXXFLAGS) $(TEST_INPUTS) -o $@

$(TEST_KV_PAGE_CACHE_TEST): runtime/engine/KvPool.cpp \
		runtime/engine/KvCache.cpp \
		dev/tests/engine/kv_page_cache_test.cpp | $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_TEST_CXXFLAGS) $(TEST_INPUTS) -o $@

$(TEST_DRAFT_CONTEXT_PLAN_TEST): runtime/model/DraftContextPlan.cpp \
		dev/benchmarks/PrefillWork.hpp \
		dev/tests/engine/draft_context_plan_test.cpp | $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_TEST_CXXFLAGS) $(TEST_INPUTS) -o $@

$(TEST_RAGGED_SCHEDULER_TEST): runtime/engine/Scheduler.cpp \
		dev/tests/engine/ragged_scheduler_test.cpp | $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_TEST_CXXFLAGS) $(TEST_INPUTS) -o $@

$(TEST_CACHE_TEST): runtime/engine/KvPool.cpp \
		runtime/engine/KvCache.cpp \
		runtime/engine/StateCache.cpp \
		runtime/engine/Cache.cpp \
		dev/tests/engine/cache_test.cpp | $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_TEST_CXXFLAGS) $(TEST_INPUTS) -o $@

$(TEST_KV_FIRST_ENGINE_TEST): runtime/engine/Scheduler.cpp \
		runtime/model/DraftContextPlan.cpp \
		runtime/engine/KvPool.cpp \
		runtime/engine/KvCache.cpp \
		runtime/engine/StateCache.cpp \
		runtime/engine/Cache.cpp \
		runtime/engine/Engine.cpp \
		dev/tests/engine/kv_first_engine_test.cpp | $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_TEST_CXXFLAGS) $(TEST_INPUTS) -o $@

$(TEST_KV_FIRST_CACHE_TEST): runtime/engine/KvPool.cpp \
		runtime/engine/KvCache.cpp \
		runtime/engine/StateCache.cpp \
		runtime/engine/Cache.cpp \
		dev/tests/engine/kv_first_cache_test.cpp | $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_TEST_CXXFLAGS) $(TEST_INPUTS) -o $@

$(TEST_ELASTIC_KV_TEST): runtime/engine/KvPool.cpp \
		dev/tests/engine/elastic_kv_pool_test.cpp | $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_TEST_CXXFLAGS) $(TEST_INPUTS) -o $@

$(TEST_PROTOCOL_TEST): runtime/engine/Protocol.cpp \
		dev/tests/engine/protocol_test.cpp | $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_TEST_CXXFLAGS) $(TEST_INPUTS) -o $@

$(TEST_NATIVE_LOOP_TEST): $(BACKEND_CONTROL_SOURCES) \
		runtime/engine/Protocol.cpp \
		runtime/engine/NativeRuntime.cpp \
		dev/tests/engine/native_engine_loop_test.cpp | $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_TEST_CXXFLAGS) $(TEST_INPUTS) -o $@

$(TEST_FD_TRANSPORT_TEST): $(BACKEND_CONTROL_SOURCES) \
		runtime/engine/Protocol.cpp \
		runtime/engine/NativeRuntime.cpp \
		runtime/engine/FdTransport.cpp \
		dev/tests/engine/native_fd_transport_test.cpp | $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_TEST_CXXFLAGS) $(TEST_INPUTS) -o $@

$(TEST_BOOTSTRAP_TEST): dev/tests/engine/runtime_bootstrap_test.mm \
		$(ENGINE_LIBRARY) $(LIB) | $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_TEST_CXXFLAGS) -fobjc-arc $< $(ENGINE_LIBRARY) \
		$(ENGINE_LINKFLAGS) -o $@

$(TEST_RESOURCES_TEST): dev/tests/engine/runtime_resources_test.mm \
		$(ENGINE_LIBRARY) $(LIB) | $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_TEST_CXXFLAGS) -fobjc-arc $< $(ENGINE_LIBRARY) \
		$(ENGINE_LINKFLAGS) -o $@

$(TEST_METRICS_TEST): dev/tests/engine/runtime_metrics_test.cpp | $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_TEST_CXXFLAGS) $(TEST_INPUTS) -o $@

$(TEST_MODEL_PACKAGE_TEST): runtime/metal/DeviceCapabilities.cpp \
		runtime/metal/MetalBackend.mm \
		$(MODEL_SOURCES) \
		$(MODEL_OPERATOR_SOURCES) \
		dev/tests/engine/model_package_test.cpp | $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_TEST_CXXFLAGS) -fobjc-arc $(TEST_INPUTS) \
		$(ENGINE_LINKFLAGS) -o $@

$(TEST_MEMORY_AUDIT_TEST): runtime/metal/DeviceCapabilities.cpp \
		runtime/engine/MemoryPlan.cpp runtime/engine/MemoryAudit.cpp \
		dev/tests/engine/memory_audit_test.cpp | $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_TEST_CXXFLAGS) $(TEST_INPUTS) -o $@

$(TEST_QWEN_STATE_TEST): runtime/metal/DeviceCapabilities.cpp \
		runtime/metal/MetalBackend.mm \
		runtime/engine/MemoryGovernor.cpp \
		runtime/ops/PageStorage.mm \
		runtime/model/WeightStore.cpp \
		runtime/model/DFlashDraft.cpp \
		$(MODEL_OPERATOR_SOURCES) \
		runtime/model/QwenState.cpp \
		dev/tests/engine/qwen_state_storage_test.mm | $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_TEST_CXXFLAGS) -fobjc-arc $(TEST_INPUTS) \
		$(ENGINE_LINKFLAGS) -o $@

$(TEST_STATUS_TEST): runtime/metal/DeviceCapabilities.cpp \
		runtime/engine/MemoryPlan.cpp runtime/engine/MemoryAudit.cpp \
		runtime/engine/Status.cpp \
		dev/tests/engine/runtime_status_test.cpp | $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_TEST_CXXFLAGS) $(TEST_INPUTS) -o $@

$(TEST_Q8_CPU_TEST): dev/tests/engine/q8_paged_kv_test.cc \
		dev/tests/engine/Q8PageFormatReference.hpp | $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_TEST_CXXFLAGS) $< -o $@

$(TEST_Q8_AIR): dev/tests/engine/q8_page_format_oracle.metal runtime/metal/abi/ExecutionGeometry.h \
		| $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(METAL) $(TEST_METALFLAGS) -c $< -o $@

$(TEST_Q8_LIB): $(TEST_Q8_AIR)
	$(RUN_CONFIGURED) $(METALLIB) $< -o $@

$(ENGINE_TEST_BUILD)/kernels/%.air: runtime/metal/kernels/%.metal \
		$(KERNEL_HEADERS) \
		| $(ENGINE_TEST_BUILD)
	@mkdir -p $(dir $@)
	$(RUN_CONFIGURED) $(METAL) $(TEST_METALFLAGS) -c $< -o $@

$(TEST_Q8_ATTENTION_LIB): $(TEST_Q8_KERNEL_AIRS) $(TEST_Q8_AIR)
	$(RUN_CONFIGURED) $(METALLIB) $(BUILD_INPUTS) -o $@

$(TEST_Q8_ATTENTION_TEST): dev/tests/engine/q8_flash_attention_metal_test.mm \
		dev/tests/engine/Q8PageFormatReference.hpp $(TEST_Q8_ATTENTION_LIB) | $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_TEST_CXXFLAGS) -fobjc-arc $< \
		$(ENGINE_LINKFLAGS) -o $@

$(TEST_Q8_PREFILL_TEST): dev/tests/engine/q8_chunked_prefill_metal_test.mm \
		dev/tests/engine/Q8PageFormatReference.hpp $(ENGINE_LIBRARY) \
		$(TEST_Q8_ATTENTION_LIB) | $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_TEST_CXXFLAGS) -fobjc-arc $< $(ENGINE_LIBRARY) \
		$(ENGINE_LINKFLAGS) -o $@

$(TEST_Q4_BATCH_TEST): runtime/metal/DeviceCapabilities.cpp \
		runtime/metal/MetalBackend.mm \
		dev/tests/engine/q4_batched_projection_metal_test.mm $(LIB) | $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_TEST_CXXFLAGS) -fobjc-arc $(TEST_INPUTS) \
		$(ENGINE_LINKFLAGS) -o $@

$(TEST_Q4_PREFILL_TEST): runtime/metal/DeviceCapabilities.cpp \
		runtime/metal/MetalBackend.mm \
		dev/tests/engine/q4_prefill_projection_metal_test.mm $(LIB) | $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_TEST_CXXFLAGS) -fobjc-arc $(TEST_INPUTS) \
		$(ENGINE_LINKFLAGS) -o $@

$(TEST_MOE_METAL_TEST): runtime/metal/DeviceCapabilities.cpp \
		runtime/metal/MetalBackend.mm runtime/model/WeightStore.cpp \
		runtime/ops/Linear.cpp runtime/ops/MoE.cpp \
		dev/tests/engine/moe_metal_test.mm $(LIB) | $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_TEST_CXXFLAGS) -fobjc-arc $(TEST_INPUTS) \
		$(ENGINE_LINKFLAGS) -o $@

$(TEST_GDN_METAL_TEST): runtime/metal/DeviceCapabilities.cpp \
		runtime/metal/MetalBackend.mm runtime/ops/GDN.cpp \
		dev/tests/engine/gdn_metal_test.mm $(LIB) | $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_TEST_CXXFLAGS) -fobjc-arc $(TEST_INPUTS) \
		$(ENGINE_LINKFLAGS) -o $@

$(TEST_OPERATOR_WORKSPACE): dev/tests/engine/operator_workspace_test.cc \
		$(ENGINE_LIBRARY) | $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_TEST_CXXFLAGS) $< $(ENGINE_LIBRARY) \
		$(ENGINE_LINKFLAGS) -o $@

$(TEST_OPERATOR_TUNING): dev/tuning/Tuning.cpp \
		dev/tests/engine/operator_tuning_test.cpp | $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_TEST_CXXFLAGS) $(TEST_INPUTS) -o $@

$(TEST_OPERATOR_MEASUREMENT): dev/tuning/Tuning.cpp dev/tuning/Measurement.cpp \
		dev/tests/engine/operator_measurement_test.cpp | $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_TEST_CXXFLAGS) $(TEST_INPUTS) -o $@

$(TEST_EXECUTION_PLANS): dev/tests/engine/execution_plans_test.cc \
		$(ENGINE_LIBRARY) | $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_TEST_CXXFLAGS) $< $(ENGINE_LIBRARY) \
		$(ENGINE_LINKFLAGS) -o $@

$(TEST_MODEL_EXECUTION_PLANS): dev/tests/engine/model_execution_plan_test.cpp \
		$(ENGINE_LIBRARY) | $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_TEST_CXXFLAGS) $< $(ENGINE_LIBRARY) \
		$(ENGINE_LINKFLAGS) -o $@

$(TEST_LINEAR_TUNING): dev/tests/engine/linear_tuning_test.cc $(TUNING_SOURCES) \
		$(ENGINE_LIBRARY) | $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_TEST_CXXFLAGS) $(TEST_INPUTS) \
		$(ENGINE_LINKFLAGS) -o $@

$(TEST_ATTENTION_TUNING): dev/tests/engine/attention_tuning_test.cc $(TUNING_SOURCES) \
		$(ENGINE_LIBRARY) | $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_TEST_CXXFLAGS) $(TEST_INPUTS) \
		$(ENGINE_LINKFLAGS) -o $@

$(TEST_DRAFT_ATTENTION_TUNING): dev/tests/engine/draft_attention_tuning_test.cc $(TUNING_SOURCES) \
		$(ENGINE_LIBRARY) | $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_TEST_CXXFLAGS) $(TEST_INPUTS) \
		$(ENGINE_LINKFLAGS) -o $@

$(TEST_MOE_TUNING): dev/tests/engine/moe_tuning_test.mm $(TUNING_SOURCES) \
		$(ENGINE_LIBRARY) | $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_TEST_CXXFLAGS) -fobjc-arc $(TEST_INPUTS) \
		$(ENGINE_LINKFLAGS) -o $@

$(TEST_TUNING_WORKLOADS): dev/tests/engine/tuning_workloads_test.cpp $(TUNING_SOURCES) \
		$(ENGINE_LIBRARY) | $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_TEST_CXXFLAGS) $(TEST_INPUTS) \
		$(ENGINE_LINKFLAGS) -o $@

$(TUNE_KERNELS): dev/tuning/tune_kernels.mm $(TUNING_SOURCES) \
		$(ENGINE_LIBRARY) $(BUILD_ID_HEADER) | $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_CXXFLAGS) -Idev -fobjc-arc -include $(BUILD_ID_HEADER) \
		$(filter-out $(BUILD_ID_HEADER),$(TEST_INPUTS)) \
		$(ENGINE_LINKFLAGS) -o $@

# Offline kernel measurement for this device and model: reports every key
# where a precompiled candidate beats the policy default in runtime/ops.
# Run on an idle host after a kernel or policy change.
.PHONY: tune-kernels
tune-kernels: preflight $(TARGET) $(TUNE_KERNELS) $(LIB)
	$(TUNE_KERNELS) $(LIB) $(MODEL_ROOT) $(TUNE_ARGS)

$(TEST_ATTENTION_PLAN): runtime/metal/DeviceCapabilities.cpp \
		runtime/metal/MetalBackend.mm runtime/ops/PagedAttention.cpp \
		dev/tests/engine/paged_attention_plan_test.mm $(LIB) | $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_TEST_CXXFLAGS) -fobjc-arc $(TEST_INPUTS) \
		$(ENGINE_LINKFLAGS) -o $@

$(TEST_LINEAR_PLAN): runtime/metal/DeviceCapabilities.cpp \
		runtime/metal/MetalBackend.mm runtime/ops/Linear.cpp \
		dev/tests/engine/linear_plan_test.mm $(LIB) | $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_TEST_CXXFLAGS) -fobjc-arc $(TEST_INPUTS) \
		$(ENGINE_LINKFLAGS) -o $@

$(TEST_DFLASH_BATCH_CONTROL_TEST): runtime/metal/DeviceCapabilities.cpp \
		runtime/metal/MetalBackend.mm \
		dev/tests/engine/dflash_batch_control_metal_test.mm $(LIB) | $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_TEST_CXXFLAGS) -fobjc-arc $(TEST_INPUTS) \
		$(ENGINE_LINKFLAGS) -o $@

$(TEST_DRAFT_ATTENTION_TEST): runtime/metal/DeviceCapabilities.cpp \
		runtime/metal/MetalBackend.mm runtime/ops/DraftAttention.cpp \
		dev/tests/engine/draft_attention_metal_test.mm $(LIB) | $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_TEST_CXXFLAGS) -fobjc-arc $(TEST_INPUTS) \
		$(ENGINE_LINKFLAGS) -o $@

$(TEST_GDN_DECODE_TEST): runtime/metal/DeviceCapabilities.cpp \
		runtime/metal/MetalBackend.mm runtime/ops/GDN.cpp \
		dev/tests/engine/gdn_decode_metal_test.mm $(LIB) | $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_TEST_CXXFLAGS) -fobjc-arc $(TEST_INPUTS) \
		$(ENGINE_LINKFLAGS) -o $@

$(TEST_DRAFT_SELECTOR_TEST): runtime/metal/DeviceCapabilities.cpp \
		runtime/metal/MetalBackend.mm runtime/ops/Sampling.cpp \
		dev/tests/engine/draft_selector_metal_test.mm $(LIB) | $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_TEST_CXXFLAGS) -fobjc-arc $(TEST_INPUTS) \
		$(ENGINE_LINKFLAGS) -o $@

$(TEST_Q4_PREFILL_PROFILE): dev/benchmarks/q4_prefill_profile.mm \
		$(ENGINE_LIBRARY) | $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_CXXFLAGS) -fobjc-arc $< $(ENGINE_LIBRARY) \
		$(ENGINE_LINKFLAGS) -o $@

$(TEST_Q4_DECODE_PROFILE): dev/benchmarks/q4_decode_profile.mm \
		$(ENGINE_LIBRARY) | $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_CXXFLAGS) -fobjc-arc $< $(ENGINE_LIBRARY) \
		$(ENGINE_LINKFLAGS) -o $@

$(TEST_Q8_METAL_TEST): dev/tests/engine/q8_paged_kv_metal_test.mm \
		dev/tests/engine/Q8PageFormatReference.hpp | $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_TEST_CXXFLAGS) -fobjc-arc $< \
		$(ENGINE_LINKFLAGS) -o $@

$(TEST_Q8_STORAGE_TEST): runtime/metal/DeviceCapabilities.cpp \
		runtime/metal/MetalBackend.mm \
		runtime/engine/MemoryGovernor.cpp \
		runtime/ops/PageStorage.mm \
		dev/tests/engine/q8_page_storage_test.mm | $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_TEST_CXXFLAGS) -fobjc-arc $(TEST_INPUTS) \
		$(ENGINE_LINKFLAGS) -o $@

$(TEST_METAL_BACKEND_AIR): dev/tests/engine/metal_backend_test.metal \
		| $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(METAL) $(TEST_METALFLAGS) -c $< -o $@

$(TEST_METAL_BACKEND_LIB): $(TEST_METAL_BACKEND_AIR)
	$(RUN_CONFIGURED) $(METALLIB) $< -o $@

$(TEST_METAL_BACKEND_TEST): runtime/metal/DeviceCapabilities.cpp \
		runtime/metal/MetalBackend.mm \
		dev/tests/engine/metal_backend_test.mm | $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_TEST_CXXFLAGS) -fobjc-arc $(TEST_INPUTS) \
		$(ENGINE_LINKFLAGS) -o $@

$(TEST_VISION_ENCODER_TEST): runtime/metal/DeviceCapabilities.cpp \
		runtime/metal/MetalBackend.mm \
		$(MODEL_SOURCES) \
		$(MODEL_OPERATOR_SOURCES) \
		runtime/ops/Vision.mm \
		dev/tests/engine/vision_encoder_test.mm | $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_TEST_CXXFLAGS) -fobjc-arc $(TEST_INPUTS) \
		$(ENGINE_LINKFLAGS) -o $@

$(TEST_MODEL_RUNTIME_ORACLE): dev/tests/engine/model_runtime_oracle_test.mm \
		$(ENGINE_LIBRARY) $(LIB) | $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_TEST_CXXFLAGS) -fobjc-arc $< \
		$(ENGINE_LIBRARY) \
		$(ENGINE_LINKFLAGS) -o $@


$(TEST_DECODE_PROFILE): dev/benchmarks/decode_profile.mm \
		$(ENGINE_LIBRARY) $(LIB) | $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_TEST_CXXFLAGS) -fobjc-arc $< \
		$(ENGINE_LIBRARY) \
		$(ENGINE_LINKFLAGS) -o $@

$(TEST_ATTENTION_SWEEP): dev/benchmarks/attention_sweep.mm \
		$(ENGINE_LIBRARY) $(LIB) | $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_TEST_CXXFLAGS) -fobjc-arc $< \
		$(ENGINE_LIBRARY) \
		$(ENGINE_LINKFLAGS) -o $@

$(TEST_BACKEND_BENCHMARK): dev/benchmarks/backend_benchmark.mm \
		dev/benchmarks/PrefillWork.hpp \
		$(ENGINE_LIBRARY) $(LIB) $(BUILD_ID_HEADER) \
		| $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_OBJCXXFLAGS) -include $(BUILD_ID_HEADER) $< \
		$(ENGINE_LIBRARY) \
		$(ENGINE_LINKFLAGS) -o $@

.PHONY: verify-build-identity
verify-build-identity: $(TARGET) $(BUILD_ID_HEADER) $(BUILD_ID_STAMP)
	$(BUILD_ID_PYTHON) $(BUILD_ID_SCRIPT) check --root . \
		--header $(BUILD_ID_HEADER) --stamp $(BUILD_ID_STAMP) \
		$(BUILD_ID_CONSTANT_ARGS) --binary $(TARGET)

# Metal kernel tests run with GPU-side bounds checking so a kernel that writes
# past host-sized scratch fails here instead of corrupting memory under the
# real model. (The API debug layer is not enabled: it misestimates the 32-row
# prefill kernels' threadgroup memory, which Metal itself reports as 32768.)
METAL_TEST_ENV := MTL_SHADER_VALIDATION=1
test-engine: test-engine-cpu test-engine-metal

test-engine-cpu: $(TEST_CPU_TARGETS) $(TEST_ATTENTION_SWEEP) $(TUNE_KERNELS)
	$(TEST_DEVICE_QUERIES)
	$(TEST_TUNING_WORKLOADS)
	$(TEST_LINEAR_PLAN) --cpu
	$(TEST_LINEAR_TUNING) --cpu
	$(TEST_ATTENTION_TUNING)
	$(TEST_DRAFT_ATTENTION_TUNING)
	$(TEST_MOE_TUNING) --cpu
	/bin/sh dev/tests/attention_sweep_cli.sh $(TEST_ATTENTION_SWEEP)
	/bin/sh dev/tests/tune_kernels_cli.sh $(TUNE_KERNELS)
	$(TEST_OPERATOR_WORKSPACE)
	$(TEST_OPERATOR_TUNING)
	$(TEST_OPERATOR_MEASUREMENT)
	$(TEST_EXECUTION_PLANS)
	$(TEST_MODEL_EXECUTION_PLANS)
	$(TEST_MEMORY_TEST)
	$(TEST_KV_PAGE_CACHE_TEST)
	$(TEST_KV_FIRST_CACHE_TEST)
	$(TEST_DRAFT_CONTEXT_PLAN_TEST)
	$(TEST_RAGGED_SCHEDULER_TEST)
	$(TEST_CACHE_TEST)
	$(TEST_KV_FIRST_ENGINE_TEST)
	$(TEST_ELASTIC_KV_TEST)
	$(TEST_PROTOCOL_TEST)
	$(TEST_NATIVE_LOOP_TEST)
	$(TEST_FD_TRANSPORT_TEST)
	$(TEST_BOOTSTRAP_TEST)
	$(TEST_METRICS_TEST)
	$(TEST_MEMORY_AUDIT_TEST)
	$(TEST_STATUS_TEST)
	$(TEST_Q8_CPU_TEST)

$(TEST_Q4_SGMATRIX_TEST): dev/tests/engine/q4_sgmatrix_metal_test.mm $(ENGINE_LIBRARY) $(LIB) | $(ENGINE_TEST_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_TEST_CXXFLAGS) -fobjc-arc $< $(ENGINE_LIBRARY) $(ENGINE_LINKFLAGS) -o $@

test-engine-metal: $(TEST_METAL_TARGETS)
	$(METAL_TEST_ENV) $(TEST_TUNING_WORKLOADS) $(LIB)
	$(METAL_TEST_ENV) $(TEST_LINEAR_TUNING) $(LIB)
	$(METAL_TEST_ENV) $(TEST_ATTENTION_TUNING) --metal $(LIB)
	$(METAL_TEST_ENV) $(TEST_DRAFT_ATTENTION_TUNING) --metal $(LIB)
	$(METAL_TEST_ENV) $(TEST_MOE_TUNING) $(LIB)
	$(METAL_TEST_ENV) $(TEST_ATTENTION_PLAN) $(LIB)
	$(METAL_TEST_ENV) $(TEST_LINEAR_PLAN) $(LIB)
	$(METAL_TEST_ENV) $(TEST_RESOURCES_TEST) $(LIB)
	$(METAL_TEST_ENV) $(TEST_MODEL_PACKAGE_TEST) $(TEST_METAL_BACKEND_LIB)
	$(METAL_TEST_ENV) $(TEST_QWEN_STATE_TEST) $(TEST_Q8_LIB)
	$(METAL_TEST_ENV) $(TEST_Q8_METAL_TEST) $(TEST_Q8_LIB)
	$(METAL_TEST_ENV) $(TEST_Q8_STORAGE_TEST) $(TEST_Q8_LIB)
	$(METAL_TEST_ENV) $(TEST_Q8_ATTENTION_TEST) $(TEST_Q8_ATTENTION_LIB)
	$(METAL_TEST_ENV) $(TEST_Q8_PREFILL_TEST) $(TEST_Q8_ATTENTION_LIB)
	$(METAL_TEST_ENV) $(TEST_Q4_BATCH_TEST) $(LIB)
	$(METAL_TEST_ENV) $(TEST_Q4_PREFILL_TEST) $(LIB)
	$(METAL_TEST_ENV) $(TEST_Q4_SGMATRIX_TEST) $(LIB)
	$(METAL_TEST_ENV) $(TEST_MOE_METAL_TEST) $(LIB)
	$(METAL_TEST_ENV) $(TEST_GDN_METAL_TEST) $(LIB)
	$(METAL_TEST_ENV) $(TEST_DFLASH_BATCH_CONTROL_TEST) $(LIB)
	$(METAL_TEST_ENV) $(TEST_DRAFT_ATTENTION_TEST) $(LIB)
	$(METAL_TEST_ENV) $(TEST_GDN_DECODE_TEST) $(LIB)
	$(METAL_TEST_ENV) $(TEST_DRAFT_SELECTOR_TEST) $(LIB)
	$(METAL_TEST_ENV) $(TEST_METAL_BACKEND_TEST) $(TEST_METAL_BACKEND_LIB)

.PHONY: test-real
VISION_FIXTURE_incoai/Qwen3.8-27B-Splash := qwen3.8-27b
VISION_FIXTURE_incoai/Qwen3.6-35B-A3B-Splash := qwen3.6-35b-a3b
VISION_FIXTURE_incoai/Qwen3.8-27B-Splash := qwen3.8-27b
VISION_FIXTURE_incoai/Qwen3.6-35B-A3B-Splash := qwen3.6-35b-a3b
test-real: preflight $(TARGET) $(TEST_MODEL_RUNTIME_ORACLE) \
		$(TEST_VISION_ENCODER_TEST) $(LIB)
	$(METAL_TEST_ENV) $(TEST_VISION_ENCODER_TEST) $(LIB) $(MODEL_ROOT) \
		dev/tests/fixtures/vision-parity/$(VISION_FIXTURE_$(MODEL))
	$(TEST_MODEL_RUNTIME_ORACLE) $(LIB) $(MODEL_ROOT)

.PHONY: benchmark-prefill benchmark-decode benchmark-backend \
	benchmark-decode-profile benchmark-attention-sweep
benchmark-prefill: all $(TEST_Q4_PREFILL_PROFILE)
	$(TEST_Q4_PREFILL_PROFILE) $(LIB)

benchmark-decode: all $(TEST_Q4_DECODE_PROFILE)
	$(TEST_Q4_DECODE_PROFILE) $(LIB)

# decode-profile replays the installed model's prefill and decode commands as
# separate dispatches; DECODE_PROFILE_ARGS passes --prompt-tokens/--cycles.
benchmark-decode-profile: preflight $(TARGET) $(TEST_DECODE_PROFILE) $(LIB)
	$(TEST_DECODE_PROFILE) $(LIB) $(MODEL_ROOT) $(DECODE_PROFILE_ARGS)

# Attention kernels alone on one layer of synthetic Q8 history across cache
# lengths; ATTENTION_SWEEP_ARGS passes --histories/--shapes/--lanes/--repeat.
benchmark-attention-sweep: $(TEST_ATTENTION_SWEEP) $(LIB)
	$(TEST_ATTENTION_SWEEP) $(LIB) $(ATTENTION_SWEEP_ARGS)

benchmark-backend: preflight $(TARGET) $(TEST_BACKEND_BENCHMARK) $(LIB)
	$(TEST_BACKEND_BENCHMARK) $(LIB) $(MODEL_ROOT)

$(TEST_BACKEND_ASAN): $(BACKEND_CONTROL_SOURCES) \
		dev/tests/engine/kv_first_engine_test.cpp | $(ENGINE_SANITIZER_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_SANITIZER_CXXFLAGS) -fsanitize=address,undefined \
		$(TEST_INPUTS) -o $@

$(TEST_BACKEND_TSAN): $(BACKEND_CONTROL_SOURCES) \
		dev/tests/engine/kv_first_engine_test.cpp | $(ENGINE_SANITIZER_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_SANITIZER_CXXFLAGS) -fsanitize=thread $(TEST_INPUTS) -o $@

$(TEST_FD_TRANSPORT_ASAN): $(BACKEND_CONTROL_SOURCES) \
		runtime/engine/Protocol.cpp \
		runtime/engine/NativeRuntime.cpp \
		runtime/engine/FdTransport.cpp \
		dev/tests/engine/native_fd_transport_test.cpp | $(ENGINE_SANITIZER_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_SANITIZER_CXXFLAGS) -fsanitize=address,undefined \
		$(TEST_INPUTS) -o $@

$(TEST_FD_TRANSPORT_TSAN): $(BACKEND_CONTROL_SOURCES) \
		runtime/engine/Protocol.cpp \
		runtime/engine/NativeRuntime.cpp \
		runtime/engine/FdTransport.cpp \
		dev/tests/engine/native_fd_transport_test.cpp | $(ENGINE_SANITIZER_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_SANITIZER_CXXFLAGS) -fsanitize=thread $(TEST_INPUTS) -o $@

$(TEST_OPERATOR_TUNING_ASAN): dev/tuning/Tuning.cpp \
		dev/tests/engine/operator_tuning_test.cpp | $(ENGINE_SANITIZER_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_SANITIZER_CXXFLAGS) -fsanitize=address,undefined \
		$(TEST_INPUTS) -o $@

$(TEST_OPERATOR_TUNING_TSAN): dev/tuning/Tuning.cpp \
		dev/tests/engine/operator_tuning_test.cpp | $(ENGINE_SANITIZER_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_SANITIZER_CXXFLAGS) -fsanitize=thread $(TEST_INPUTS) -o $@

$(TEST_OPERATOR_MEASUREMENT_ASAN): dev/tuning/Tuning.cpp dev/tuning/Measurement.cpp \
		dev/tests/engine/operator_measurement_test.cpp | $(ENGINE_SANITIZER_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_SANITIZER_CXXFLAGS) -fsanitize=address,undefined \
		$(TEST_INPUTS) -o $@

$(TEST_OPERATOR_MEASUREMENT_TSAN): dev/tuning/Tuning.cpp dev/tuning/Measurement.cpp \
		dev/tests/engine/operator_measurement_test.cpp | $(ENGINE_SANITIZER_BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_SANITIZER_CXXFLAGS) -fsanitize=thread $(TEST_INPUTS) -o $@

.PHONY: test-sanitizers
test-sanitizers: $(TEST_BACKEND_ASAN) $(TEST_BACKEND_TSAN) \
		$(TEST_FD_TRANSPORT_ASAN) $(TEST_FD_TRANSPORT_TSAN) \
		$(TEST_OPERATOR_TUNING_ASAN) $(TEST_OPERATOR_TUNING_TSAN) \
		$(TEST_OPERATOR_MEASUREMENT_ASAN) $(TEST_OPERATOR_MEASUREMENT_TSAN)
	ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 \
		UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
		$(TEST_BACKEND_ASAN)
	TSAN_OPTIONS=halt_on_error=1 $(TEST_BACKEND_TSAN)
	ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 \
		UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
		$(TEST_FD_TRANSPORT_ASAN)
	TSAN_OPTIONS=halt_on_error=1 $(TEST_FD_TRANSPORT_TSAN)
	ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 \
		UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
		$(TEST_OPERATOR_TUNING_ASAN)
	TSAN_OPTIONS=halt_on_error=1 $(TEST_OPERATOR_TUNING_TSAN)
	ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 \
		UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
		$(TEST_OPERATOR_MEASUREMENT_ASAN)
	TSAN_OPTIONS=halt_on_error=1 $(TEST_OPERATOR_MEASUREMENT_TSAN)

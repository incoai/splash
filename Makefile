XCRUN := xcrun
CXX := $(XCRUN) -sdk macosx clang++
METAL := $(XCRUN) -sdk macosx metal
METALLIB := $(XCRUN) -sdk macosx metallib
AR := $(XCRUN) -sdk macosx ar
SYSTEM_NAME := $(shell uname -s)
SYSTEM_ARCH := $(shell uname -m)
VENV := .venv
PYTHON = $(VENV)/bin/python
# Holds the hash of the requirements the environment was last installed from.
VENV_STAMP = $(VENV)/.requirements-installed
INSTALL_LOCK = $(VENV).install.lock
REQUIREMENTS := install/requirements.txt
PYTHON_CANDIDATES := python3.13 python3 python3.12 python3.14
BUILD_ID_PYTHON ?= python3
SPLASH_MAKEFILE := $(abspath $(firstword $(MAKEFILE_LIST)))
MODEL_INSTALL = $(PYTHON) install/models.py
MODEL ?=
MODEL_ROOT := install/models/$(MODEL)

BUILD := build
TARGET := $(BUILD)/splash
METAL_BUILD := $(BUILD)/metal
# Production kernels are grouped by execution phase under
# runtime/metal/kernels/{prefill,decode,shared}; every .metal file there is
# compiled into the metallib. Templates both phases instantiate live in
# kernels/common/, the host/shader contract in runtime/metal/abi/.
PRODUCTION_KERNEL_SOURCES := $(sort $(wildcard \
	runtime/metal/kernels/prefill/*.metal \
	runtime/metal/kernels/decode/*.metal \
	runtime/metal/kernels/shared/*.metal))
PRODUCTION_KERNEL_NAMES := \
	$(patsubst runtime/metal/kernels/%.metal,%,$(PRODUCTION_KERNEL_SOURCES))
PRODUCTION_AIRS := $(addprefix $(METAL_BUILD)/, \
	$(addsuffix .air,$(PRODUCTION_KERNEL_NAMES)))
KERNEL_HEADERS := $(sort $(wildcard runtime/metal/abi/*.h \
	runtime/metal/kernels/common/*.h))
# Placement-sparse support became queryable in macOS 26.4
# (MTLDevice.supportsPlacementSparse). The engine refuses older systems at
# startup; every binary and metallib records the same floor.
MACOS_MIN_VERSION := 26.4
MACOS_TARGET_FLAG := -mmacosx-version-min=$(MACOS_MIN_VERSION)
PROD_METALFLAGS := -std=metal4.0 -O3 -Wall -Wextra -Werror -Iruntime \
	$(MACOS_TARGET_FLAG)
ENGINE_CXXFLAGS := -std=c++20 -O3 -Wall -Wextra -Werror -Iruntime \
	$(MACOS_TARGET_FLAG)
ENGINE_OBJCXXFLAGS := $(ENGINE_CXXFLAGS) -fobjc-arc
LIB := $(BUILD)/splash.metallib
.PHONY: all clean force-build-identity install _install \
	install-environment _install-environment \
	platform-check model-selection preflight serve verify-models

all: $(TARGET)

install: model-selection platform-check
	@/usr/bin/lockf -k "$(INSTALL_LOCK)" $(MAKE) --no-print-directory \
		-f "$(SPLASH_MAKEFILE)" _install

_install: model-selection _install-environment
	$(MODEL_INSTALL) --model "$(MODEL)" prepare

model-selection:
	@test -n "$(MODEL)" || { \
		echo "error: set MODEL to a supported full Hugging Face repository ID" >&2; \
		exit 1; \
	}

platform-check:
	@test "$(SYSTEM_NAME)" = Darwin && test "$(SYSTEM_ARCH)" = arm64 || { \
		echo "error: Splash requires an Apple Silicon Mac" >&2; \
		exit 1; \
	}
	@command -v "$(XCRUN)" >/dev/null 2>&1 || { \
		echo "error: Xcode with Metal tools is required" >&2; \
		exit 1; \
	}
	@for tool in clang++ metal metallib; do \
		"$(XCRUN)" -sdk macosx -f "$$tool" >/dev/null 2>&1 || { \
			echo "error: Xcode with Metal tools is required (missing $$tool)" >&2; \
			exit 1; \
		}; \
	done

install-environment:
	@/usr/bin/lockf -k "$(INSTALL_LOCK)" $(MAKE) --no-print-directory \
		-f "$(SPLASH_MAKEFILE)" _install-environment

_install-environment:
	@set -eu; \
	if test -f "$(VENV)/pyvenv.cfg" && test -x "$(PYTHON)" \
		&& "$(PYTHON)" -c 'import sys; raise SystemExit(not ((3, 12) <= sys.version_info[:2] < (3, 15) and sys.prefix != sys.base_prefix))' \
			>/dev/null 2>&1 \
		&& "$(PYTHON)" -m pip --version >/dev/null 2>&1; then \
		:; \
	else \
		bootstrap=; \
		for candidate in $(PYTHON_CANDIDATES); do \
			command -v "$$candidate" >/dev/null 2>&1 || continue; \
			base=$$("$$candidate" -c 'import sys; print(getattr(sys, "_base_executable", sys.executable))') \
				|| continue; \
			"$$base" -c 'import sys; raise SystemExit(not ((3, 12) <= sys.version_info[:2] < (3, 15)))' \
				>/dev/null 2>&1 || continue; \
			bootstrap=$$base; \
			break; \
		done; \
		test -n "$$bootstrap" || { \
			echo "error: Python 3.12, 3.13, or 3.14 is required" >&2; \
			exit 1; \
		}; \
		if test -L "$(VENV)"; then \
			echo "error: refusing to rebuild symlinked environment $(VENV)" >&2; \
			exit 1; \
		fi; \
		if test -e "$(VENV)"; then \
			test -d "$(VENV)" && test -f "$(VENV)/pyvenv.cfg" || { \
				echo "error: refusing to clear non-venv path $(VENV)" >&2; \
				exit 1; \
			}; \
			echo "Rebuilding unsupported or broken environment in $(VENV)..."; \
			"$$bootstrap" -m venv --clear "$(VENV)"; \
		else \
			"$$bootstrap" -m venv "$(VENV)"; \
		fi; \
	fi; \
	hash=$$(shasum -a 256 "$(REQUIREMENTS)" | awk '{print $$1}'); \
	if test "$$(cat "$(VENV_STAMP)" 2>/dev/null)" = "$$hash"; then \
		exit 0; \
	fi; \
	rm -f "$(VENV_STAMP)"; \
	"$(PYTHON)" -m pip install --only-binary=:all: -r "$(REQUIREMENTS)"; \
	"$(PYTHON)" -m pip check; \
	tmp=$$(mktemp "$(VENV_STAMP).tmp.XXXXXX"); \
	trap 'rm -f "$$tmp"' EXIT INT TERM; \
	echo "$$hash" > "$$tmp"; \
	mv "$$tmp" "$(VENV_STAMP)"; \
	trap - EXIT INT TERM

preflight: model-selection
	@test -x $(PYTHON) || { \
		echo "error: Splash is not installed; run 'make install MODEL=$(MODEL)' first" >&2; \
		exit 1; \
	}
	@$(MODEL_INSTALL) --model "$(MODEL)" verify
	@$(PYTHON) -m pip check >/dev/null
	@TRANSFORMERS_VERBOSITY=error $(PYTHON) -c 'import server.server'

verify-models: preflight
	@$(MODEL_INSTALL) --model "$(MODEL)" verify --full

serve: preflight $(TARGET)
	./splash serve --model "$(MODEL)"

$(BUILD):
	mkdir -p $(BUILD)

$(METAL_BUILD): | $(BUILD)
	mkdir -p $@

$(METAL_BUILD)/%.air: runtime/metal/kernels/%.metal $(KERNEL_HEADERS) \
		| $(METAL_BUILD)
	@mkdir -p $(dir $@)
	$(RUN_CONFIGURED) $(METAL) $(PROD_METALFLAGS) -c $< -o $@

$(LIB): $(PRODUCTION_AIRS)
	$(RUN_CONFIGURED) $(METALLIB) $(BUILD_INPUTS) -o $@

ENGINE_BUILD := $(BUILD)/engine
ENGINE_LIBRARY := $(ENGINE_BUILD)/libsplash.a
ENGINE_LINKFLAGS := -framework Foundation -framework Metal -framework IOKit
ENGINE_DEPFLAGS := -MMD -MP
# Configuration belongs to each successful output, not to a shared timestamp:
# macOS make can treat a new stamp and an old binary in the same second as equal.
BUILD_CONFIG_TOOL := dev/tools/build_config.py
BUILD_RULES_DIGEST := $(shell shasum -a 256 Makefile dev/native.mk \
	$(BUILD_CONFIG_TOOL) | shasum -a 256 | cut -c1-16)
shell-quote = '$(subst ','"'"',$(1))'
CONFIG_DIGEST := $(shell printf '%s\0' $(BUILD_RULES_DIGEST) \
	$(call shell-quote,$(CXX)) $(call shell-quote,$(ENGINE_CXXFLAGS)) \
	$(call shell-quote,$(ENGINE_OBJCXXFLAGS)) $(call shell-quote,$(ENGINE_DEPFLAGS)) \
	$(call shell-quote,$(ENGINE_LINKFLAGS)) $(call shell-quote,$(AR)) \
	$(call shell-quote,$(METAL)) $(call shell-quote,$(PROD_METALFLAGS)) \
	$(call shell-quote,$(METALLIB)) | shasum -a 256 | cut -c1-16)
RUN_CONFIGURED = $(BUILD_ID_PYTHON) $(BUILD_CONFIG_TOOL) record \
	--output $(call shell-quote,$@) --config $(BUILD_CONFIG) --
BUILD_INPUTS = $(filter-out force-build-config,$^)
BUILD_ID_SCRIPT := dev/tools/build_identity.py
BUILD_ID_HEADER := $(ENGINE_BUILD)/BuildIdentity.hpp
BUILD_ID_STAMP := $(ENGINE_BUILD)/build-identity.json
PRODUCTION_ENGINE_INPUTS := $(sort $(shell find runtime -type f \
	\( -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.h' \
		-o -name '*.hpp' -o -name '*.m' -o -name '*.mm' \) \
	-print))
BUILD_ID_CONSTANT_ARGS = \
	--constant 'runtime=qwen-hybrid-dflash8' \
	--constant 'engine_cxxflags=$(ENGINE_CXXFLAGS)' \
	--constant 'engine_objcxxflags=$(ENGINE_OBJCXXFLAGS)' \
	--constant 'engine_linkflags=$(ENGINE_LINKFLAGS)' \
	--constant 'production_metalflags=$(PROD_METALFLAGS)'
ENGINE_MAIN_OBJECT := $(ENGINE_BUILD)/main.o
ENGINE_METAL_RUNTIME_OBJECT := $(ENGINE_BUILD)/metal/MetalBackend.o
ENGINE_CPP_SOURCES := \
	runtime/ops/DraftAttention.cpp \
	runtime/ops/Embedding.cpp \
	runtime/ops/ExecutionPlans.cpp \
	runtime/ops/GDN.cpp \
	runtime/ops/Linear.cpp \
	runtime/ops/MoE.cpp \
	runtime/ops/Normalization.cpp \
	runtime/ops/PagedAttention.cpp \
	runtime/ops/RoPE.cpp \
	runtime/ops/Sampling.cpp \
	runtime/metal/DeviceCapabilities.cpp \
	runtime/engine/MemoryPlan.cpp \
	runtime/engine/Scheduler.cpp \
	runtime/engine/Cache.cpp \
	runtime/engine/Engine.cpp \
	runtime/engine/MemoryGovernor.cpp \
	runtime/engine/KvPool.cpp \
	runtime/engine/KvCache.cpp \
	runtime/engine/StateCache.cpp \
	runtime/model/DraftContextPlan.cpp \
	runtime/engine/Protocol.cpp \
	runtime/engine/NativeRuntime.cpp \
	runtime/engine/FdTransport.cpp \
	runtime/engine/MemoryAudit.cpp \
	runtime/engine/Status.cpp \
	runtime/model/WeightStore.cpp \
	runtime/model/Qwen3_6Moe.cpp \
	runtime/model/Qwen3_8.cpp \
	runtime/model/QwenVision.cpp \
	runtime/model/QwenTarget.cpp \
	runtime/model/DFlashDraft.cpp \
	runtime/model/ModelFactory.cpp \
	runtime/model/QwenState.cpp
ENGINE_MM_SOURCES := \
	runtime/model/ModelDescriptor.mm \
	runtime/model/Runtime.mm \
	runtime/model/RuntimeArenas.mm \
	runtime/ops/Vision.mm \
	runtime/ops/PageStorage.mm \
	runtime/engine/RuntimeResources.mm \
	runtime/engine/Bootstrap.mm
ENGINE_OBJECTS := \
	$(patsubst runtime/%.cpp,$(ENGINE_BUILD)/%.o,$(ENGINE_CPP_SOURCES)) \
	$(patsubst runtime/%.mm,$(ENGINE_BUILD)/%.o,$(ENGINE_MM_SOURCES)) \
	$(ENGINE_METAL_RUNTIME_OBJECT)
PRODUCTION_CONFIG_TARGETS := $(ENGINE_OBJECTS) $(ENGINE_MAIN_OBJECT) \
	$(ENGINE_LIBRARY) $(PRODUCTION_AIRS) $(LIB) $(TARGET)
ENGINE_DEPFILES := $(ENGINE_OBJECTS:.o=.d) $(ENGINE_MAIN_OBJECT:.o=.d)

-include $(ENGINE_DEPFILES)

force-build-identity:

$(ENGINE_BUILD): | $(BUILD)
	mkdir -p $@

$(BUILD_ID_STAMP): force-build-identity | $(ENGINE_BUILD)
	@$(BUILD_ID_PYTHON) $(BUILD_ID_SCRIPT) write --root . \
		--header $(BUILD_ID_HEADER) --stamp $(BUILD_ID_STAMP) \
		$(BUILD_ID_CONSTANT_ARGS) >/dev/null

$(BUILD_ID_HEADER): $(BUILD_ID_STAMP)
	@:

$(ENGINE_BUILD)/%.o: runtime/%.cpp
	@mkdir -p $(dir $@)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_CXXFLAGS) $(ENGINE_DEPFLAGS) -c $< -o $@

$(ENGINE_BUILD)/%.o: runtime/%.mm
	@mkdir -p $(dir $@)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_OBJCXXFLAGS) $(ENGINE_DEPFLAGS) -c $< -o $@

$(ENGINE_METAL_RUNTIME_OBJECT): runtime/metal/MetalBackend.mm
	@mkdir -p $(dir $@)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_OBJCXXFLAGS) $(ENGINE_DEPFLAGS) -c $< -o $@

$(ENGINE_MAIN_OBJECT): runtime/main.mm $(BUILD_ID_HEADER)
	@mkdir -p $(dir $@)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_OBJCXXFLAGS) $(ENGINE_DEPFLAGS) \
		-include $(BUILD_ID_HEADER) -c $< -o $@

$(ENGINE_LIBRARY): $(ENGINE_OBJECTS)
	@mkdir -p $(dir $@)
	$(RM) $@
	$(RUN_CONFIGURED) $(AR) rcs $@ $(BUILD_INPUTS)

$(TARGET): $(ENGINE_MAIN_OBJECT) $(ENGINE_LIBRARY) $(LIB) \
		| $(BUILD)
	$(RUN_CONFIGURED) $(CXX) $(ENGINE_OBJCXXFLAGS) $(ENGINE_MAIN_OBJECT) $(ENGINE_LIBRARY) \
		$(ENGINE_LINKFLAGS) -o $@

clean:
	rm -rf $(BUILD)

-include dev/Makefile

# The dev include extends these output sets. Check each set in one read-only
# process; a mismatched output has a real force dependency even within one
# timestamp tick. Only its successful recipe updates its .config record.
.PHONY: force-build-config
force-build-config:

# Removing a wildcard input removes its timestamp dependency too. Record the
# input names with the affected Metal outputs: header changes recompile AIRs,
# while source-set changes only relink the production metallib.
KERNEL_HEADER_NAMES_DIGEST := $(shell printf '%s\0' $(sort $(KERNEL_HEADERS)) \
	| shasum -a 256 | cut -c1-16)
KERNEL_SOURCE_NAMES_DIGEST := $(shell printf '%s\0' $(sort $(PRODUCTION_KERNEL_SOURCES)) \
	| shasum -a 256 | cut -c1-16)
PRODUCTION_AIR_CONFIG := $(CONFIG_DIGEST)-$(KERNEL_HEADER_NAMES_DIGEST)
PRODUCTION_LIB_CONFIG := $(PRODUCTION_AIR_CONFIG)-$(KERNEL_SOURCE_NAMES_DIGEST)
TEST_KERNEL_CONFIG := $(TEST_CONFIG_DIGEST)-$(KERNEL_HEADER_NAMES_DIGEST)
TEST_KERNEL_CONFIG_TARGETS := $(TEST_Q8_KERNEL_AIRS) \
	$(TEST_Q8_ATTENTION_LIB)
PRODUCTION_CONFIG_TARGETS := $(filter-out $(PRODUCTION_AIRS) $(LIB),$(PRODUCTION_CONFIG_TARGETS))
TEST_CONFIG_TARGETS := $(filter-out $(TEST_KERNEL_CONFIG_TARGETS),$(TEST_CONFIG_TARGETS))

$(PRODUCTION_CONFIG_TARGETS): BUILD_CONFIG := $(CONFIG_DIGEST)
$(TEST_CONFIG_TARGETS): BUILD_CONFIG := $(TEST_CONFIG_DIGEST)
$(SANITIZER_CONFIG_TARGETS): BUILD_CONFIG := $(SANITIZER_CONFIG_DIGEST)
$(PRODUCTION_AIRS): BUILD_CONFIG := $(PRODUCTION_AIR_CONFIG)
$(LIB): BUILD_CONFIG := $(PRODUCTION_LIB_CONFIG)
$(TEST_KERNEL_CONFIG_TARGETS): BUILD_CONFIG := $(TEST_KERNEL_CONFIG)
STALE_CONFIG_TARGETS := $(shell $(BUILD_ID_PYTHON) $(BUILD_CONFIG_TOOL) stale \
	--config $(CONFIG_DIGEST) $(PRODUCTION_CONFIG_TARGETS)) \
	$(shell $(BUILD_ID_PYTHON) $(BUILD_CONFIG_TOOL) stale \
	--config $(TEST_CONFIG_DIGEST) $(TEST_CONFIG_TARGETS)) \
	$(shell $(BUILD_ID_PYTHON) $(BUILD_CONFIG_TOOL) stale \
	--config $(SANITIZER_CONFIG_DIGEST) $(SANITIZER_CONFIG_TARGETS)) \
	$(shell $(BUILD_ID_PYTHON) $(BUILD_CONFIG_TOOL) stale \
	--config $(PRODUCTION_AIR_CONFIG) $(PRODUCTION_AIRS)) \
	$(shell $(BUILD_ID_PYTHON) $(BUILD_CONFIG_TOOL) stale \
	--config $(PRODUCTION_LIB_CONFIG) $(LIB)) \
	$(shell $(BUILD_ID_PYTHON) $(BUILD_CONFIG_TOOL) stale \
	--config $(TEST_KERNEL_CONFIG) $(TEST_KERNEL_CONFIG_TARGETS))
ifneq ($(strip $(STALE_CONFIG_TARGETS)),)
$(STALE_CONFIG_TARGETS): force-build-config
endif

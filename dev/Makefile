# Holds the hash of the development requirements last installed.
DEV_STAMP = $(VENV)/.dev-requirements-installed
RELEASE_VERSION ?= $(shell git describe --tags --always)

.PHONY: package package-bottle package-check
package: install-environment all
	$(PYTHON) dev/tools/package.py --version "$(RELEASE_VERSION)" \
		--macos-min "$(MACOS_MIN_VERSION)"

package-bottle:
	$(PYTHON) dev/tools/package_brew.py build --version "$(RELEASE_VERSION)"

# Both Homebrew targets use a temporary tap and remove their installation.
package-check:
	$(PYTHON) dev/tools/package_brew.py check --version "$(RELEASE_VERSION)"

# Uploads the built RELEASE_VERSION archive, its checksum, the tester installer
# and the `latest` pointer to the private Hugging Face repo testers install
# from. Uses the developer's own `hf auth login`.
TEST_RELEASE_REPO ?=
.PHONY: publish-test
publish-test:
	@test -n "$(TEST_RELEASE_REPO)" || { echo "set TEST_RELEASE_REPO=owner/repo"; exit 1; }
	$(PYTHON) dev/tools/publish_test.py --version "$(RELEASE_VERSION)" \
		--repo "$(TEST_RELEASE_REPO)"

PYTHON_TEST_PATTERN ?= test_*.py
# Comma-separated subset of claude,opencode,codex,hermes,pi for the real client
# gates on a machine that lacks one of them; empty runs all five.
AGENT_CLIENTS ?=
AGENT_CLIENT_ARGS = $(if $(AGENT_CLIENTS),--clients $(AGENT_CLIENTS),)
# smoke: implement and resume; complete adds compaction and cancellation.
AGENT_SCENARIO ?= complete
AGENT_ARGS = $(MODEL_ARGS) --model-root "$(MODEL_ROOT)" --scenario $(AGENT_SCENARIO) \
	$(AGENT_CLIENT_ARGS)
HTTP_SMOKE_ARGS ?=

# Define native targets before aggregate prerequisites expand.
-include dev/native.mk

.PHONY: agent-preflight architecture-check check check-source \
	check-native-engine check-native-cpu check-native-metal check-native-build \
	check-python \
	install-development _install-development release-check test \
	test-agent-real test-http-real test-performance-real test-release-real \
	test-python test-real verify-models \
	test-engine test-engine-cpu test-engine-metal test-sanitizers

# Decided by the stamp's content, as the environment is: a rebuilt
# environment has no stamp, and under -j make may read the stamp's time
# before install-environment rebuilds it.
install-development: install-environment
	@$(ANNOUNCE_INSTALL_WAIT)
	@/usr/bin/lockf -k "$(INSTALL_LOCK)" $(MAKE) --no-print-directory \
		-f "$(SPLASH_MAKEFILE)" _install-development

_install-development:
	@set -eu; \
	hash=$$(shasum -a 256 dev/requirements.txt | awk '{print $$1}'); \
	if test "$$(cat "$(DEV_STAMP)" 2>/dev/null)" = "$$hash"; then \
		exit 0; \
	fi; \
	"$(PYTHON)" -m pip install -r dev/requirements.txt; \
	echo "$$hash" > "$(DEV_STAMP)"

# Build production and every native test and tool, and run the native tests
# that do not require model weights.
check-native-engine: check-native-cpu check-native-metal check-native-build

check-native-cpu: platform-check $(TARGET) test-engine-cpu

check-native-metal: platform-check $(TARGET) test-engine-metal

# Build every native test, benchmark and tool, the outputs TEST_CONFIG_TARGETS
# and PRODUCTION_FLAG_TOOLS name, and run none: hosted CI has no GPU the Metal
# tests support and no model for the real-model tests and profiles, and builds
# them so that none breaks unnoticed.
check-native-build: platform-check $(TARGET) $(TEST_CONFIG_TARGETS) \
	$(PRODUCTION_FLAG_TOOLS)

# Every Python test, each in the directory of what it tests (dev/tests/server,
# install or tools). Discovering dev/tests keeps this gate complete as tests
# are added; discovered from the checkout root (-t .), a module has the package
# name it is imported by, so none loads twice. The installer tests check
# configurations with the engine.
test-python: install-development $(TARGET)
	$(PYTHON) -m unittest discover -s dev/tests -t . -p '$(PYTHON_TEST_PATTERN)'

check-python: install-development test-python
	$(PYTHON) -m compileall -q server install/*.py \
		dev/tools dev/tests dev/benchmarks
	$(PYTHON) -m ruff check --config dev/ruff.toml server \
		install/*.py dev/tools dev/tests dev/benchmarks
	$(PYTHON) -m ruff format --check --config dev/ruff.toml server \
		install/*.py dev/tools dev/tests dev/benchmarks

# It parses the server's sources, which need the environment's Python
# (3.12-3.14); BUILD_ID_PYTHON, the build tools' python3, may be Xcode's 3.9.
architecture-check: install-environment
	$(PYTHON) dev/tools/check_architecture.py

check: check-source check-native-engine check-python

check-source: architecture-check
	@git diff --check
	@empty_tree=$$(git hash-object -t tree /dev/null); \
		git diff --check $$empty_tree HEAD

test: test-engine test-python

test-http-real: preflight $(TARGET) verify-build-identity
	TRANSFORMERS_VERBOSITY=error $(PYTHON) dev/tests/smoke_real.py \
		--model "$(MODEL)" --model-root "$(MODEL_ROOT)" $(HTTP_SMOKE_ARGS)

# Finds the selected clients and runs test_clients.py's tests of the installed
# OpenCode, Codex and Pi among them, which need no model.
agent-preflight: model-selection install-development
	$(PYTHON) dev/tests/agent_real.py $(AGENT_ARGS) --preflight-only

test-agent-real: preflight $(TARGET) agent-preflight verify-build-identity
	TRANSFORMERS_VERBOSITY=error $(PYTHON) dev/tests/agent_real.py $(AGENT_ARGS)

# The release path loads the model once for generic HTTP and all five official
# clients. The individual targets remain useful while diagnosing one surface.
test-release-real: preflight $(TARGET) agent-preflight verify-build-identity
	TRANSFORMERS_VERBOSITY=error $(PYTHON) dev/tests/agent_real.py $(AGENT_ARGS) \
		--http-smoke

# BASELINE is a checkout whose build/ holds another build's splash.metallib,
# engine-tests/backend-benchmark and, for a build that loads the weights into
# memory, engine-tests/weight-digests: this build is compared with it on this
# machine in ABBA order, and so are their weight images
# (dev/benchmarks/backend_regression.py). Without it, the release scenarios
# characterize this build alone. ANE_FFN_SHARE, in [0, 1), is the prefill
# FFN's Neural Engine share every benchmark that takes --ane-ffn-share runs
# instead of calibrating one (0: the GPU alone).
BASELINE ?=
ANE_FFN_SHARE ?=
ANE_FFN_SHARE_ARGS = $(if $(ANE_FFN_SHARE),--ane-ffn-share "$(ANE_FFN_SHARE)",)
test-performance-real: preflight $(TARGET) $(TEST_BACKEND_BENCHMARK) $(LIB) \
		$(if $(BASELINE),$(WEIGHT_DIGESTS))
	@mkdir -p "$(MODEL_RESULTS)"
	@$(BUILD_ID_PYTHON) $(BUILD_ID_SCRIPT) check --root . \
		--header $(BUILD_ID_HEADER) --stamp $(BUILD_ID_STAMP) \
		$(BUILD_ID_CONSTANT_ARGS) --binary $(TEST_BACKEND_BENCHMARK)
ifneq ($(BASELINE),)
	$(PYTHON) -m dev.benchmarks.backend_regression --baseline "$(BASELINE)" \
		--model-root "$(MODEL_ROOT)" --output-dir "$(MODEL_RESULTS)" \
		$(ANE_FFN_SHARE_ARGS)
else
	@$(TEST_BACKEND_BENCHMARK) $(LIB) "$(MODEL_ROOT)" \
		--scenario decode,partial,short --samples 3 $(ANE_FFN_SHARE_ARGS) \
		--progress "$(MODEL_RESULTS)/backend-benchmark-progress.jsonl" \
		> "$(MODEL_RESULTS)/backend-benchmark.json" \
		&& $(PYTHON) -m json.tool "$(MODEL_RESULTS)/backend-benchmark.json" >/dev/null
	@echo "native backend characterization: PASS ($(MODEL_RESULTS)/backend-benchmark.json)"
endif

# The installer's restarts without the Hub, a full source hash and the
# digests of the weight images a load of the installation holds
# (DEVELOPMENT.md, Release check).
verify-models: preflight $(TARGET) $(WEIGHT_DIGESTS) $(LIB)
	@$(PYTHON) dev/tools/installer_restarts.py $(MODEL_ARGS)
	@mkdir -p "$(MODEL_RESULTS)"
	@$(WEIGHT_DIGESTS) $(LIB) "$(MODEL_ROOT)" > "$(MODEL_RESULTS)/weights.json.tmp"
	@mv "$(MODEL_RESULTS)/weights.json.tmp" "$(MODEL_RESULTS)/weights.json"
	@echo "weight images: $(MODEL_RESULTS)/weights.json"

# One model on this machine against BASELINE, a retained build that loads it
# (DEVELOPMENT.md, Release check); make check and test-sanitizers run once per
# source identity, check-native-metal and the agent clients once per machine.
release-check: preflight
	@test -n "$(BASELINE)" || { \
		echo "error: release-check compares with BASELINE, a retained build that loads the model" >&2; \
		exit 1; \
	}
	@test -z "$$(git status --porcelain)" || { \
		echo "error: release checkout is not clean" >&2; \
		git status --short >&2; \
		exit 1; \
	}
	$(MAKE) --no-print-directory verify-build-identity
	$(MAKE) --no-print-directory test-real
	$(MAKE) --no-print-directory verify-models
	$(MAKE) --no-print-directory test-http-real
	$(MAKE) --no-print-directory test-http-real \
		HTTP_SMOKE_ARGS="--persistent-cache --max-cache-disk 8G"
	$(MAKE) --no-print-directory test-performance-real
	@test -z "$$(git status --porcelain)" || { \
		echo "error: release checkout is not clean" >&2; \
		git status --short >&2; \
		exit 1; \
	}

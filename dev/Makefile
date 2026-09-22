DEV_STAMP := .venv/.dev-requirements-installed
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

ENGINE_PYTHON_TEST_PATTERN ?= test_*.py
# Comma-separated subset of claude,opencode,codex,hermes for the real client
# gates on a machine that lacks one of them; empty runs all four.
AGENT_CLIENTS ?=
AGENT_CLIENT_ARGS = $(if $(AGENT_CLIENTS),--clients $(AGENT_CLIENTS),)
HTTP_SMOKE_ARGS ?=

# Define native targets before aggregate prerequisites expand.
-include dev/native.mk
-include dev/flash.mk

.PHONY: agent-preflight architecture-check check check-source \
	check-native-engine check-native-cpu check-native-metal check-python-engine \
	install-development release-check test test-agent-real test-http-real \
	test-performance-real test-release-real \
	test-models test-python test-real \
	test-engine test-engine-cpu test-engine-metal test-sanitizers

install-development: install-environment $(DEV_STAMP)

$(DEV_STAMP): dev/requirements.txt | install-environment
	$(PYTHON) -m pip install -r dev/requirements.txt
	@shasum -a 256 dev/requirements.txt | awk '{print $$1}' > $(DEV_STAMP)

# Build production and run native tests that do not require model weights.
check-native-engine: check-native-cpu check-native-metal

check-native-cpu: platform-check $(TARGET) test-engine-cpu

check-native-metal: platform-check $(TARGET) test-engine-metal

# Discovering dev/tests/engine keeps this gate complete as new current runtime Python tests are
# added.  dev/tests/test_models.py validates the installer and model manifests
# without downloading or loading the real weights.
test-python: install-development
	$(PYTHON) -m unittest discover -s dev/tests/engine -p '$(ENGINE_PYTHON_TEST_PATTERN)'
	$(PYTHON) -m unittest dev/tests/test_server.py

test-models: install-development
	$(PYTHON) -m unittest dev/tests/test_models.py

check-python-engine: install-development test-python test-models
	$(PYTHON) -m compileall -q server install/*.py \
		dev/tools dev/tests dev/benchmarks
	$(PYTHON) -m ruff check --config dev/ruff.toml server \
		install/*.py dev/tools dev/tests dev/benchmarks
	$(PYTHON) -m ruff format --check --config dev/ruff.toml server \
		install/*.py dev/tools dev/tests dev/benchmarks

architecture-check:
	$(BUILD_ID_PYTHON) dev/tools/check_architecture.py

check: check-source check-native-engine check-python-engine

check-source: architecture-check
	@git diff --check
	@empty_tree=$$(git hash-object -t tree /dev/null); \
		git diff --check $$empty_tree HEAD

test: test-engine test-python test-models

test-http-real: preflight $(TARGET) verify-build-identity
	TRANSFORMERS_VERBOSITY=error $(PYTHON) dev/tests/smoke_real.py \
		--model "$(MODEL)" $(HTTP_SMOKE_ARGS)

agent-preflight: model-selection install-development
	$(PYTHON) dev/tests/agent_real.py --model "$(MODEL)" --preflight-only $(AGENT_CLIENT_ARGS)

test-agent-real: preflight $(TARGET) agent-preflight verify-build-identity
	TRANSFORMERS_VERBOSITY=error $(PYTHON) dev/tests/agent_real.py --model "$(MODEL)" \
		$(AGENT_CLIENT_ARGS)

# The release path loads the model once for generic HTTP and all four official
# clients. The individual targets remain useful while diagnosing one surface.
test-release-real: preflight $(TARGET) agent-preflight verify-build-identity
	TRANSFORMERS_VERBOSITY=error $(PYTHON) dev/tests/agent_real.py \
		--model "$(MODEL)" --http-smoke $(AGENT_CLIENT_ARGS)

# Characterizes and validates one source identity. This target deliberately
# does not claim an old/new regression result; that requires two retained
# benchmark binaries and same-machine alternating execution.
test-performance-real: preflight $(TARGET) $(TEST_BACKEND_BENCHMARK) $(LIB)
	@mkdir -p build/release
	@$(BUILD_ID_PYTHON) $(BUILD_ID_SCRIPT) check --root . \
		--header $(BUILD_ID_HEADER) --stamp $(BUILD_ID_STAMP) \
		$(BUILD_ID_CONSTANT_ARGS) --binary $(TEST_BACKEND_BENCHMARK)
	@set +e; \
		$(TEST_BACKEND_BENCHMARK) $(LIB) $(MODEL_ROOT) --samples 5 \
			--progress build/release/backend-benchmark-progress.jsonl \
			> build/release/backend-benchmark.json; \
		status=$$?; \
		$(PYTHON) -m json.tool build/release/backend-benchmark.json >/dev/null \
			|| exit 1; \
		test $$status -eq 0 || exit $$status
	@echo "native backend characterization: PASS (build/release/backend-benchmark.json)"

# One final source identity: model-free correctness, sanitizers, real-model
# correctness, one shared HTTP/four-client server, then native performance.
release-check: check preflight
	@test -z "$$(git status --porcelain)" || { \
		echo "error: release checkout is not clean" >&2; \
		git status --short >&2; \
		exit 1; \
	}
	$(MAKE) --no-print-directory verify-models
	$(MAKE) --no-print-directory verify-build-identity
	$(MAKE) --no-print-directory test-sanitizers
	$(MAKE) --no-print-directory test-real
	$(MAKE) --no-print-directory test-release-real
	$(MAKE) --no-print-directory test-performance-real
	@test -z "$$(git status --porcelain)" || { \
		echo "error: release checkout is not clean" >&2; \
		git status --short >&2; \
		exit 1; \
	}

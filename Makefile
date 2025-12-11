# AI-Tools Makefile
# Usage: make [target] [SERVICE=name]
#
# Docker builds ALWAYS run checks first (lint, format, test)
# Examples:
#   make build                    # Build all services (with checks)
#   make build SERVICE=vosk-asr   # Build specific service (with checks)
#   make build-quick              # Build without checks (not recommended)

.PHONY: build build-quick up down logs test lint format check install tray tray-quick

# Default Python from .venv
VENV := .venv/Scripts
PYTHON := $(VENV)/python.exe
PIP := $(PYTHON) -m pip

# Service to build (optional, builds all if not specified)
SERVICE ?=

# Paths
LIVE_CAPTIONS := apps/live-captions

# Plain text output (Windows compatible)

#------------------------------------------------------------------------------
# Docker service targets
#------------------------------------------------------------------------------

## build: Run checks then build Docker services
## Usage: make build [SERVICE=parakeet-asr|vosk-asr|whisper-asr|...]
build: check
	@echo ""
	@echo "=== DOCKER BUILD ==="
ifdef SERVICE
	@echo "Building service: $(SERVICE)"
	docker compose build $(SERVICE)
else
	@echo "Building all services..."
	docker compose build
endif
	@echo "[OK] Build complete"

## build-quick: Build Docker services without checks (not recommended)
build-quick:
ifdef SERVICE
	docker compose build $(SERVICE)
else
	docker compose build
endif

## up: Start all services
up:
	docker compose up -d

## down: Stop all services
down:
	docker compose down

## logs: Tail logs from all services
logs:
	docker compose logs -f

#------------------------------------------------------------------------------
# Live Captions tray app targets
#------------------------------------------------------------------------------

## tray: Run checks then build Live Captions tray app (PyInstaller)
tray: check
	@echo "=== Building Live Captions tray app ==="
	cd $(LIVE_CAPTIONS) && $(PYTHON) -m PyInstaller "Live Captions Tray.spec" --noconfirm
	@echo "[OK] Tray app built: $(LIVE_CAPTIONS)/dist/"

## tray-quick: Build tray app without checks
tray-quick:
	cd $(LIVE_CAPTIONS) && $(PYTHON) -m PyInstaller "Live Captions Tray.spec" --noconfirm

#------------------------------------------------------------------------------
# Code quality targets
#------------------------------------------------------------------------------

## check: Run all checks (lint, format, test)
check: lint format test
	@echo ""
	@echo "=== ALL CHECKS PASSED ==="

## lint: Run ruff linter with auto-fix
lint:
	@echo "[1/3] Ruff (lint)..."
	@$(PYTHON) -m ruff check --fix . --quiet
	@echo "[OK] Lint passed"

## format: Run black formatter
format:
	@echo "[2/3] Black (format)..."
	@$(PYTHON) -m black . --quiet
	@echo "[OK] Format passed"

## test: Run unit tests
test:
	@echo "[3/3] Pytest (tests)..."
	@$(PYTHON) -m pytest apps/ shared/ services/ --tb=short -q
	@echo "[OK] Tests passed"

## test-all: Run all tests including service tests (requires fastapi)
test-all:
	@echo "Running all tests..."
	$(PYTHON) -m pytest apps/ shared/ services/ --tb=short -q --ignore=services/audio-notes
	@echo "[OK] All tests passed"

## coverage: Run tests with coverage and generate HTML report
coverage:
	@echo "=== Running tests with coverage ==="
	$(PYTHON) -m pytest apps/ shared/ --cov=apps --cov=shared --cov-report=html --cov-report=term-missing
	@echo "[OK] Coverage report: htmlcov/index.html"

## coverage-all: Full coverage including services
coverage-all:
	@echo "=== Running all tests with coverage ==="
	$(PYTHON) -m pytest apps/ shared/ services/ --cov=apps --cov=shared --cov=services --cov-report=html --cov-report=term-missing --ignore=services/audio-notes
	@echo "[OK] Coverage report: htmlcov/index.html"

## test-e2e: Run E2E tests
## Usage: make test-e2e [T=path]
## Examples:
##   make test-e2e                              # Run all E2E tests
##   make test-e2e T=test_asr_services.py       # Run specific file
##   make test-e2e T=TestVoskASR                # Run specific class
##   make test-e2e T=test_health_endpoint       # Run tests matching name
T ?=
test-e2e:
	@echo "=== E2E Tests ==="
ifeq ($(T),)
	$(PYTHON) -m pytest integration/e2e/ -v --tb=short
else
	$(PYTHON) -m pytest integration/e2e/ -v --tb=short -k "$(T)"
endif
	@echo "[OK] E2E tests passed"

#------------------------------------------------------------------------------
# Setup targets
#------------------------------------------------------------------------------

## install: Install all dependencies in .venv
install:
	$(PIP) install -r requirements.txt
	$(PIP) install -r $(LIVE_CAPTIONS)/requirements.txt
	@echo "[OK] Dependencies installed"

## install-dev: Install dev tools (ruff, black, pytest, etc.)
install-dev:
	$(PIP) install -r requirements-dev.txt
	@echo "[OK] Dev tools installed"

## install-all: Install all dependencies including dev tools
install-all: install install-dev
	@echo "[OK] All dependencies installed"

#------------------------------------------------------------------------------
# Help
#------------------------------------------------------------------------------

## help: Show this help message
help:
	@echo "Available targets:"
	@grep -E '^##' $(MAKEFILE_LIST) | sed 's/## /  /'

.PHONY: help build release test clean install uninstall format format-check clippy check run ci all

.DEFAULT_GOAL := help

BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
NC := \033[0m

help: ## Show this help message
	@echo "$(BLUE)Available targets:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2}'

all: format clippy test build ## Run format, clippy, tests, and build

build: ## Build debug binary
	@echo "$(BLUE)Building debug binary...$(NC)"
	cargo build

release: ## Build optimized release binary
	@echo "$(BLUE)Building release binary...$(NC)"
	cargo build --release
	@echo "$(GREEN)Release binary: target/release/emji$(NC)"

test: ## Run all tests
	@echo "$(BLUE)Running tests...$(NC)"
	cargo test

test-verbose: ## Run tests with output
	@echo "$(BLUE)Running tests (verbose)...$(NC)"
	cargo test -- --nocapture

check: ## Quick check if code compiles (faster than build)
	@echo "$(BLUE)Checking code...$(NC)"
	cargo check

format: ## Format code with rustfmt
	@echo "$(BLUE)Formatting code...$(NC)"
	cargo fmt

format-check: ## Check if code is formatted correctly
	@echo "$(BLUE)Checking code formatting...$(NC)"
	cargo fmt -- --check

clippy: ## Run clippy lints
	@echo "$(BLUE)Running clippy...$(NC)"
	cargo clippy --all-targets --all-features -- -D warnings

clippy-fix: ## Run clippy and automatically fix issues
	@echo "$(BLUE)Running clippy with auto-fix...$(NC)"
	cargo clippy --fix --allow-dirty --allow-staged

clean: ## Remove build artifacts
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	cargo clean

install: release ## Install binary to cargo bin directory
	@echo "$(BLUE)Installing emji...$(NC)"
	cargo install --path .
	@echo "$(GREEN)Installed! Run 'emji' to use$(NC)"

uninstall: ## Uninstall binary from cargo bin directory
	@echo "$(BLUE)Uninstalling emji...$(NC)"
	cargo uninstall emji
	@echo "$(GREEN)Uninstalled!$(NC)"

run: ## Run the program (usage: make run ARGS="your query here")
	@echo "$(BLUE)Running emji...$(NC)"
	cargo run -- $(ARGS)

run-release: release ## Run the release binary (usage: make run-release ARGS="your query")
	@echo "$(BLUE)Running emji (release)...$(NC)"
	./target/release/emji $(ARGS)

ci: format-check clippy test ## Run all CI checks (format, clippy, tests)
	@echo "$(GREEN)All CI checks passed!$(NC)"

bench: ## Run benchmarks (if any exist)
	@echo "$(BLUE)Running benchmarks...$(NC)"
	cargo bench

doc: ## Generate and open documentation
	@echo "$(BLUE)Generating documentation...$(NC)"
	cargo doc --open

update: ## Update dependencies
	@echo "$(BLUE)Updating dependencies...$(NC)"
	cargo update

outdated: ## Check for outdated dependencies (requires cargo-outdated)
	@echo "$(BLUE)Checking for outdated dependencies...$(NC)"
	@command -v cargo-outdated >/dev/null 2>&1 || { echo "$(YELLOW)Installing cargo-outdated...$(NC)"; cargo install cargo-outdated; }
	cargo outdated

bloat: ## Analyze binary size (requires cargo-bloat)
	@echo "$(BLUE)Analyzing binary size...$(NC)"
	@command -v cargo-bloat >/dev/null 2>&1 || { echo "$(YELLOW)Installing cargo-bloat...$(NC)"; cargo install cargo-bloat; }
	cargo bloat --release

watch: ## Watch for changes and rebuild (requires cargo-watch)
	@echo "$(BLUE)Watching for changes...$(NC)"
	@command -v cargo-watch >/dev/null 2>&1 || { echo "$(YELLOW)Installing cargo-watch...$(NC)"; cargo install cargo-watch; }
	cargo watch -x check -x test

coverage: ## Generate code coverage report (requires cargo-tarpaulin)
	@echo "$(BLUE)Generating coverage report...$(NC)"
	@command -v cargo-tarpaulin >/dev/null 2>&1 || { echo "$(YELLOW)Installing cargo-tarpaulin...$(NC)"; cargo install cargo-tarpaulin; }
	cargo tarpaulin --out Html

build-linux: ## Build for Linux x86_64
	@echo "$(BLUE)Building for Linux x86_64...$(NC)"
	cargo build --release --target x86_64-unknown-linux-gnu

build-macos-intel: ## Build for macOS Intel
	@echo "$(BLUE)Building for macOS Intel...$(NC)"
	cargo build --release --target x86_64-apple-darwin

build-macos-arm: ## Build for macOS Apple Silicon
	@echo "$(BLUE)Building for macOS ARM...$(NC)"
	cargo build --release --target aarch64-apple-darwin

build-all: build-linux build-macos-intel build-macos-arm ## Build for all platforms
	@echo "$(GREEN)Built for all platforms!$(NC)"

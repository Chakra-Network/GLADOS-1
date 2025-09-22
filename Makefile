VENV_DIR = .venv
UV = uv
UV_PATH = $(HOME)/.local/bin/uv


check_uv:
	@if [ -x "$(UV_PATH)" ]; then \
		echo ""; \
	elif command -v $(UV) >/dev/null 2>&1; then \
		echo "uv found in PATH. Proceeding..."; \
	else \
		echo "uv not found. Installing..."; \
		if [ -f /etc/os-release ] && grep -q "ubuntu" /etc/os-release; then \
			sudo mkdir -p ~/.config/fish/conf.d; \
			sudo chmod 755 ~/.config/fish/conf.d; \
		else \
			echo "Detected non-Ubuntu system. Proceeding with installation..."; \
		fi; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	fi

install: check_uv
	@echo "Activating virtual environment and installing requirements..."
	@$(UV) sync
	@$(UV) pip install --upgrade pip
	@$(UV) pip install -r requirements.txt
	@echo "Setting file descriptor limit..."
	@ulimit -n 1000000 || echo "Warning: Could not set file descriptor limit"
	@echo "Requirements installed successfully"

train: check_uv
	@echo "Fine-tuning the model with grounding dataset..."
	@COMMIT_HASH=$$(git rev-parse --short HEAD); \
	echo "Using commit hash: $$COMMIT_HASH"; \
	$(UV) run python -m code.train --dataset_type grounding --commit_hash $$COMMIT_HASH

train_state_transition: check_uv
	@echo "Fine-tuning the model with state transition dataset..."
	@COMMIT_HASH=$$(git rev-parse --short HEAD); \
	echo "Using commit hash: $$COMMIT_HASH"; \
	$(UV) run python -m code.train --dataset_type state_transition --commit_hash $$COMMIT_HASH

train_compliance: check_uv
	@echo "Fine-tuning the model with compliance dataset..."
	@COMMIT_HASH=$$(git rev-parse --short HEAD); \
	echo "Using commit hash: $$COMMIT_HASH"; \
	$(UV) run python -m code.train --dataset_type compliance --commit_hash $$COMMIT_HASH

test_sample: check_uv
	@echo "Running tests..."
	@$(UV) pip install -e .
	@$(UV) run python -m tests.test_grounding_converter --sample
	@$(UV) run python -m tests.test_state_transition_converter --sample

test_all: check_uv
	@echo "Running tests..."
	@$(UV) pip install -e .
	@$(UV) run python -m tests.test_grounding_converter
	@$(UV) run python -m tests.test_state_transition_converter

test_image_transformation: check_uv
	@echo "Running tests..."
	@$(UV) pip install -e .
	@$(UV) run python -m tests.test_image_transformation

test_sequential_pairs: check_uv
	@echo "Running tests..."
	@$(UV) pip install -e .
	@$(UV) run python -m tests.test_sequential_pairs

build_script $(script_name):
	cd code/scripts && bash build_script.sh $(script_name)

build_image_downloader:
	$(MAKE) build_script script_name=image_downloader
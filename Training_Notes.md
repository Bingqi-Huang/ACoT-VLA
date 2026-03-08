1. Prepare the environment
    Make sure your extracted dataset root exists under ~/Datasets/lerobot/Reasoning2Action-Sim or set:
    ```bash
    export ACOT_CHALLENGE_DATA_ROOT=/your/Reasoning2Action-Sim
    ```
    If you want a different base init checkpoint, set:
    ```bash
    export ACOT_CHALLENGE_INIT_WEIGHTS=/path/to/base/params
    ```

2. Compute generalist norm stats
    This must run before training the new configs:
    
    ```bash 
    UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/compute_norm_stats.py \
    --config-name acot_challenge_generalist_lora_all
    ```
    This should write stats into the assets dir used by the config.
3. Run a short generalist debug train
    First smoke test the new accumulation path:
    ```bash
    DEBUG_MODE=true UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train.py \
    --config-name acot_challenge_generalist_lora_all \
    --exp_name generalist_debug \
    --overwrite
    ```
    Check:
    - training starts
    - no OOM
    - loss logs appear
    - checkpoint is written

4. Run a real generalist train
    If debug is clean:
    ```bash
    bash scripts/train.sh acot_challenge_generalist_lora_all generalist_v1
    ```
    Or directly:
    ```bash
    UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train.py \
    --config-name acot_challenge_generalist_lora_all \
    --exp_name generalist_v1
    ```
    Wait until you have a usable checkpoint, ideally the target final step.

5. Point specialists at the real generalist checkpoint
    Export the checkpoint path:
    
    ```bash
    export ACOT_CHALLENGE_GENERALIST_WEIGHTS=./checkpoints/acot_challenge_generalist_lora_all/generalist_v1/50000/params
    ```
    
    If your best step is not 50000, use the real step directory.
6. Run one specialist debug job first
    Example:
    ```bash
    DEBUG_MODE=true UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train.py \
    --config-name acot_specialist_pour_workpiece \
    --exp_name specialist_debug \
    --overwrite
    ```
    Check the same things as the generalist debug run.

7. Train the specialists
    One by one or in parallel:
    ```bash
        bash scripts/train.sh acot_specialist_pour_workpiece specialist_v1
        bash scripts/train.sh acot_specialist_open_door specialist_v1
        bash scripts/train.sh acot_specialist_scoop_popcorn specialist_v1
        bash scripts/train.sh acot_specialist_hold_pot specialist_v1
        bash scripts/train.sh acot_specialist_place_block specialist_v1
        bash scripts/train.sh acot_specialist_take_wrong_item specialist_v1
        bash scripts/train.sh acot_specialist_stock_shelf specialist_v1
        bash scripts/train.sh acot_specialist_sorting specialist_v1
        bash scripts/train.sh acot_specialist_clean_desktop specialist_v1
    ```
8. Extract adapters

    Create the adapter directory:
    ```bash
    mkdir -p adapters
    ```
    Extract the fallback generalist adapter:
    ```bash
    UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/extract_adapter.py \
    --checkpoint ./checkpoints/acot_challenge_generalist_lora_all/generalist_v1/50000 \
    --output ./adapters/_default.npz
    ```
    Extract each specialist adapter:
    ```bash
    UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/extract_adapter.py \
    --checkpoint ./checkpoints/acot_specialist_pour_workpiece/specialist_v1/10000 \
    --output ./adapters/pour_workpiece.npz
    ```
    Repeat for the others.

9. Smoke test routed serving locally
    Set the runtime paths:
    ```bash
    export ACOT_ROUTED_CONFIG=acot_challenge_generalist_lora_all
    export ACOT_ROUTED_BASE_CHECKPOINT=./checkpoints/acot_challenge_generalist_lora_all/generalist_v1/50000
    export ACOT_ROUTED_ADAPTER_DIR=./adapters
    ```
    Start the server:
    ```bash
    bash scripts/server_routed.sh 0 8999
    ```
    Check health:
    ```bash
    curl -L http://127.0.0.1:8999/healthz
    ```

10. Test real routing behavior
    Verify at least:

    - task_name="pour_workpiece"
    - task_name="sorting_packages"

    - unknown task name fallback
    Confirm:
    - server responds
    - action shape is valid
    - waist behavior is correct for sorting
    - fallback uses _default

11. Validate Docker startup
    Build with routed startup:
    ```bash
    SERVER_SCRIPT=./scripts/server_routed.sh docker build -f scripts/docker/serve_policy.Dockerfile -t acot-routed .
    ```
    Run it:
    ```bash
    docker run --rm -p 8999:8999 \
    -e SERVER_SCRIPT=./scripts/server_routed.sh \
    -e ACOT_ROUTED_CONFIG=acot_challenge_generalist_lora_all \
    -e ACOT_ROUTED_BASE_CHECKPOINT=./checkpoints/acot_challenge_generalist_lora_all/generalist_v1/50000 \
    -e ACOT_ROUTED_ADAPTER_DIR=./adapters \
    acot-routed
    ```
    - Recheck /healthz.
12. Before submission
    Confirm:
    - checkpoint exists in image
    - adapters exist in image
    - server auto-starts
    - server listens on 8999
    - websocket path works
    Then update:
        ```bash 
        AGENTS/status.md
        AGENTS/handoff.md
        AGENTS/experiments.md
        ```
    Recommended immediate next 3 commands
    ```bash
    export ACOT_CHALLENGE_DATA_ROOT=~/Datasets/lerobot/Reasoning2Action-Sim
    UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/compute_norm_stats.py --config-name acot_challenge_generalist_lora_all
    DEBUG_MODE=true UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/train.py --config-name acot_challenge_generalist_lora_all --exp_name generalist_debug --overwrite
    ```
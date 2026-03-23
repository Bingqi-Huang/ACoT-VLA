FROM sim-icra-registry.cn-beijing.cr.aliyuncs.com/icra-admin/openpi_server:latest

WORKDIR /submission

COPY pyproject.toml uv.lock .python-version LICENSE README.md /submission/
COPY packages /submission/packages
COPY scripts /submission/scripts
COPY src /submission/src
COPY third_party /submission/third_party

# Generalist checkpoint is the routing base.
# The checkpoint directory must contain:
#   params/                                        <- model weights
#   assets/reasoning2action_sim_generalist/        <- norm stats (copy from project assets/)
# Place your chosen step folder directly at checkpoint/generalist_continued/
# before running docker build.
COPY checkpoint/generalist_continued /submission/checkpoint/generalist_continued

# Specialist adapter NPZs extracted with:
#   uv run python scripts/extract_adapter.py \
#     --checkpoint checkpoints/<config>/<exp>/<step> \
#     --output adapters/<task>.npz \
#     --include-dual-ae
# Expected files: clean_desktop.npz, stock_shelf.npz, place_block.npz,
#                 sorting.npz, pour_workpiece.npz
COPY adapters /submission/adapters

# Build-time dependency install so runtime serving does not need uv sync/install.
RUN uv sync --frozen --no-dev
# uv.lock pins jax-cuda13 for RTX 5090 (local dev). Test server has RTX 4090 with CUDA 12.x
# drivers — override to cuda12 wheels so the container works on the submission server.
RUN .venv/bin/pip install --upgrade "jax[cuda12]" -q

ENV PATH=/submission/.venv/bin:${PATH}

RUN chmod +x /submission/scripts/server_submit_routed.sh

ENV OPENPI_DATA_HOME=/root/.cache/openpi
ENV PYTHONPATH=/submission:/submission/src:/submission/packages/openpi-client/src
# Config must match the architecture of the base checkpoint.
ENV ACOT_ROUTED_CONFIG=acot_challenge_generalist_continued
ENV ACOT_ROUTED_BASE_CHECKPOINT=/submission/checkpoint/generalist_continued
ENV ACOT_ROUTED_ADAPTER_DIR=/submission/adapters
# No ACOT_ROUTED_SPECIALIST_NORM_STATS_PATH: all tasks share the same norm stats
# embedded in the generalist checkpoint under assets/reasoning2action_sim_generalist/.

RUN uv run --no-sync python -c "from openpi.models.tokenizer import PaligemmaTokenizer; PaligemmaTokenizer()"

EXPOSE 8999

CMD ["/bin/bash", "-lc", "/submission/scripts/server_submit_routed.sh 0 8999"]

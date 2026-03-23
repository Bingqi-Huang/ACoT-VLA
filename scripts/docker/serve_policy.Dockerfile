FROM sim-icra-registry.cn-beijing.cr.aliyuncs.com/icra-admin/openpi_server:latest

WORKDIR /submission

ENV UV_PROJECT_ENVIRONMENT=/submission/.venv

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

# Build-time dependency install so runtime serving does not need uv sync/install.
RUN GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-dev --no-install-package lerobot

ENV PATH=${UV_PROJECT_ENVIRONMENT}/bin:${PATH}

# uv.lock pins jax[cuda13]==0.7.2 for local RTX 5090 development.
# The submission server uses CUDA 12.x, so replace the locked CUDA13 JAX stack
# with the matching CUDA12 0.7.2 packages to avoid mixed PJRT plugins.
RUN uv pip uninstall --python /submission/.venv/bin/python jax-cuda13-plugin jax-cuda13-pjrt jax jaxlib \
    && uv pip install --python /submission/.venv/bin/python "jax[cuda12]==0.7.2" -q

RUN chmod +x /submission/scripts/server_submit_routed.sh

ENV OPENPI_DATA_HOME=/root/.cache/openpi
ENV PYTHONPATH=/submission:/submission/src:/submission/packages/openpi-client/src
# Config must match the architecture of the base checkpoint.
ENV ACOT_ROUTED_CONFIG=acot_challenge_generalist_continued
ENV ACOT_ROUTED_BASE_CHECKPOINT=/submission/checkpoint/generalist_continued
# No adapters in this generalist-only submission — ACOT_ROUTED_ADAPTER_DIR intentionally unset.
# No ACOT_ROUTED_SPECIALIST_NORM_STATS_PATH: norm stats embedded in checkpoint dir.

RUN uv run --no-sync python -c "from openpi.models.tokenizer import PaligemmaTokenizer; PaligemmaTokenizer()"

EXPOSE 8999

CMD ["/bin/bash", "-lc", "/submission/scripts/server_submit_routed.sh 0 8999"]

FROM sim-icra-registry.cn-beijing.cr.aliyuncs.com/icra-admin/openpi_server:latest
WORKDIR /submission

ENV UV_PROJECT_ENVIRONMENT=/submission/.venv

COPY pyproject.toml uv.lock .python-version LICENSE README.md /submission/
COPY packages /submission/packages
COPY scripts /submission/scripts
COPY src /submission/src
COPY third_party /submission/third_party
# Checkpoint directory: place your chosen step under checkpoint/generalist_continued_augumented/
# Expected layout:
#   checkpoint/generalist_continued_augumented/params/
#   checkpoint/generalist_continued_augumented/assets/reasoning2action_sim_generalist/
COPY checkpoint/generalist_continued_augumented /submission/checkpoint/generalist_continued_augumented

# Build-time dependency install so runtime serving does not need uv sync/install.
RUN GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-dev --no-install-package lerobot

ENV PATH=${UV_PROJECT_ENVIRONMENT}/bin:${PATH}

RUN chmod +x /submission/scripts/server_submit_generalist_v3.sh

ENV OPENPI_DATA_HOME=/root/.cache/openpi
ENV PYTHONPATH=/submission:/submission/src:/submission/packages/openpi-client/src
ENV ACOT_SERVE_CONFIG=acot_challenge_generalist_continued_reweighted
ENV ACOT_SERVE_CHECKPOINT=/submission/checkpoint/generalist_continued_augumented

RUN uv run --no-sync python -c "from openpi.models.tokenizer import PaligemmaTokenizer; PaligemmaTokenizer()"
EXPOSE 8999

CMD ["/bin/bash", "-lc", "/submission/scripts/server_submit_generalist_v3.sh 0 8999"]

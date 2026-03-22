FROM sim-icra-registry.cn-beijing.cr.aliyuncs.com/icra-admin/openpi_server:latest

WORKDIR /submission

COPY pyproject.toml uv.lock .python-version LICENSE README.md /submission/
COPY packages /submission/packages
COPY scripts /submission/scripts
COPY src /submission/src
COPY third_party /submission/third_party
# Checkpoint directory: place your chosen step under checkpoint/generalist_continued/
# Expected layout:
#   checkpoint/generalist_continued/params/
#   checkpoint/generalist_continued/assets/reasoning2action_sim_generalist/
COPY checkpoint/generalist_continued /submission/checkpoint/generalist_continued

# Build-time dependency install so runtime serving does not need uv sync/install.
RUN uv sync --frozen --no-dev

ENV PATH=/submission/.venv/bin:${PATH}

RUN chmod +x /submission/scripts/server_submit_generalist_v2.sh

ENV OPENPI_DATA_HOME=/root/.cache/openpi
ENV PYTHONPATH=/submission:/submission/src:/submission/packages/openpi-client/src
ENV ACOT_SERVE_CONFIG=acot_challenge_generalist_continued
ENV ACOT_SERVE_CHECKPOINT=/submission/checkpoint/generalist_continued

RUN uv run --no-sync python -c "from openpi.models.tokenizer import PaligemmaTokenizer; PaligemmaTokenizer()"

EXPOSE 8999

CMD ["/bin/bash", "-lc", "/submission/scripts/server_submit_generalist_v2.sh 0 8999"]

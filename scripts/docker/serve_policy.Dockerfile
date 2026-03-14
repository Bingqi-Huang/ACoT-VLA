FROM sim-icra-registry.cn-beijing.cr.aliyuncs.com/icra-admin/openpi_server:latest

WORKDIR /submission

COPY pyproject.toml uv.lock .python-version LICENSE README.md /submission/
COPY packages /submission/packages
COPY scripts /submission/scripts
COPY src /submission/src
COPY third_party /submission/third_party
COPY checkpoint /submission/checkpoint
COPY adapters /submission/adapters

# Build-time dependency install so runtime serving does not need uv sync/install.
RUN uv sync --frozen --no-dev

ENV PATH=/submission/.venv/bin:${PATH}

RUN chmod +x /submission/scripts/server_submit_routed.sh

ENV OPENPI_DATA_HOME=/root/.cache/openpi
ENV PYTHONPATH=/submission:/submission/src:/submission/packages/openpi-client/src
ENV ACOT_ROUTED_CONFIG=acot_challenge_lora_conservative
ENV ACOT_ROUTED_BASE_CHECKPOINT=/submission/checkpoint/baseline/30000
ENV ACOT_ROUTED_ADAPTER_DIR=/submission/adapters

RUN uv run --no-sync python -c "from openpi.models.tokenizer import PaligemmaTokenizer; PaligemmaTokenizer()"

EXPOSE 8999

CMD ["/bin/bash", "-lc", "/submission/scripts/server_submit_routed.sh 0 8999"]

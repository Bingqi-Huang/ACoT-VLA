FROM sim-icra-registry.cn-beijing.cr.aliyuncs.com/icra-admin/openpi_server:latest

WORKDIR /submission

COPY pyproject.toml uv.lock .python-version LICENSE /submission/
COPY packages /submission/packages
COPY scripts /submission/scripts
COPY src /submission/src
COPY third_party /submission/third_party
COPY checkpoint /submission/checkpoint

RUN chmod +x /submission/scripts/server_submit_generalist.sh

ENV OPENPI_DATA_HOME=/root/.cache/openpi
ENV ACOT_SERVE_CONFIG=acot_challenge_generalist_lora_generalist
ENV ACOT_SERVE_CHECKPOINT=/submission/checkpoint/generalists-v1-10000
ENV PYTHONPATH=/submission:/submission/src:/submission/packages/openpi-client/src

RUN python -c "from openpi.models.tokenizer import PaligemmaTokenizer; PaligemmaTokenizer()"

EXPOSE 8999

CMD ["/bin/bash", "-lc", "/submission/scripts/server_submit_generalist.sh 0 8999"]

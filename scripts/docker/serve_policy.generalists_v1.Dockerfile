FROM sim-icra-registry.cn-beijing.cr.aliyuncs.com/icra-admin/openpi_server:latest

WORKDIR /app
COPY . /app

ENV OPENPI_DATA_HOME=/root/.cache/openpi
ENV ACOT_SERVE_CONFIG=acot_challenge_generalist_lora_generalist
ENV ACOT_SERVE_CHECKPOINT=/app/checkpoint/generalists-v1-10000

RUN /.venv/bin/python3 -c "from openpi.models.tokenizer import PaligemmaTokenizer; PaligemmaTokenizer()"

EXPOSE 8999

CMD ["/bin/bash", "-lc", "./scripts/server_submit_generalist.sh 0 8999"]

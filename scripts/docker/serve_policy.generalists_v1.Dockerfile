FROM sim-icra-registry.cn-beijing.cr.aliyuncs.com/icra-admin/openpi_server:latest

WORKDIR /app
COPY . /app

ENV ACOT_SERVE_CONFIG=acot_challenge_generalist_lora_generalist
ENV ACOT_SERVE_CHECKPOINT=/app/checkpoint/generalists_v1_bs96_step5000

EXPOSE 8999

CMD ["/bin/bash", "-lc", "./scripts/server_submit_generalist.sh 0 8999"]

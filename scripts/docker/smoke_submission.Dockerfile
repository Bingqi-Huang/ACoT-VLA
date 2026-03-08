FROM sim-icra-registry.cn-beijing.cr.aliyuncs.com/icra-admin/openpi_server:latest

WORKDIR /root/openpi

COPY . /root/openpi

RUN chmod +x scripts/server_smoke_submission.sh

CMD ["/bin/bash", "-lc", "./scripts/server_smoke_submission.sh"]

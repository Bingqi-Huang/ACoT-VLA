FROM registry.agibot.com/genie-sim/openpi_server:latest
COPY . .
CMD /bin/bash -c "${SERVER_SCRIPT:-./scripts/server.sh} 0 8999"

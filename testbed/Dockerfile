FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends \
        iperf3 iproute2 ca-certificates procps ethtool \
    && rm -rf /var/lib/apt/lists/*
COPY shape.sh /shape.sh
RUN chmod +x /shape.sh

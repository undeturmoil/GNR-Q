FROM vllm/vllm-openai:v0.30.0

RUN uv pip install --system --no-cache-dir "datasets==5.0.1" && \
    uv pip install --system --no-cache-dir --no-deps "torchao==0.18.0"

WORKDIR /workspace/GNR-Q
ENV PYTHONPATH=/workspace/GNR-Q/src

ENTRYPOINT ["/bin/bash"]

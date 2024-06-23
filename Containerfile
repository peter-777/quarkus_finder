FROM registry.access.redhat.com/ubi9/ubi-minimal:latest AS prepare

WORKDIR /app

RUN microdnf install -y python3.12 && \
    microdnf clean all

RUN python3.12 -m ensurepip && \
    python3.12 -m pip install --upgrade pip

COPY pyproject.toml /app

RUN python3.12 -m pip install --no-cache-dir poetry && \
    poetry self add poetry-plugin-export && \
    poetry config virtualenvs.in-project true && \
    poetry install

FROM registry.access.redhat.com/ubi9/ubi-minimal:latest 

WORKDIR /app

RUN microdnf install -y python3.12 && \
    microdnf clean all

ENV PATH="/app/.venv/bin:$PATH"

COPY --from=prepare /app/.venv /app/.venv
COPY quarkus_finder /app/

EXPOSE 8501

ENTRYPOINT [ "streamlit", "run" ]
CMD [ "main.py", "--server.port=8501", "--server.address=0.0.0.0" ]

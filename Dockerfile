ARG DEPENDENCY_IMAGE=zhixun-contest:a-acetaldehyde-retrieval
FROM ${DEPENDENCY_IMAGE} AS dependencies
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 \
    BASELINE_CPU_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
WORKDIR /app
COPY --from=dependencies /usr/local/lib/python3.12/site-packages/ /usr/local/lib/python3.12/site-packages/
COPY src/baselines/ /app/baselines/
COPY src/harness.py src/contest_submission.py src/artifact_predict.py src/train_models.py src/bearing_sampling.py src/profiles.json /app/
COPY results/release/A/ /opt/retrain/models/A/
COPY results/release/B/ /opt/retrain/models/B/
COPY results/release/C/ /opt/retrain/models/C/
RUN python3 -c "import numpy; from importlib.metadata import version; assert numpy.__version__ == '2.5.4'; assert version('contest-harness') == '1.0.4'" \
 && python3 /app/harness.py describe
ENTRYPOINT ["python3", "/app/harness.py"]
CMD ["describe"]

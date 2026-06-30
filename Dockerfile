FROM python:3.11-slim

WORKDIR Orbital-Attitude-Control-Agent

COPY pyproject.toml .
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

COPY configs ./configs/

CMD ["python3", "-m", "src.main"]

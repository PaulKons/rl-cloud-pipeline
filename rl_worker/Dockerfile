FROM python:3.12

WORKDIR /app
COPY requirements.txt .
# Basic OS deps (usually enough; keep minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*



# Upgrade pip first
RUN pip install --no-cache-dir --upgrade pip

# Install CPU torch from PyTorch index, then the rest
# (This avoids "no matching distribution" for torch==2.9.1+cpu)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.9.1+cpu

# Now install the rest (will not reinstall torch)
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code + model
COPY worker.py evaluate.py rl_env.py ./
COPY ppo_pioneer_multiinput2.zip ./

CMD ["python", "worker.py"]

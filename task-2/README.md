# Task 2

A FastAPI-based Retrieval-Augmented Generation (RAG) service that combines document retrieval with text generation.

## Step 1:

1. Create a conda environment with the requirements.txt file

TIP: Check [this example](https://github.com/ServerlessLLM/ServerlessLLM/blob/main/docs/stable/getting_started/slurm_setup.md) for how to use slurm to create a conda environment.

```bash
conda create -n rag python=3.10 -y
conda activate rag
```

```bash
git clone https://github.com/ed-aisys/edin-mls-25-spring.git
cd edin-mls-25-spring/task-2
pip install -r requirements.txt
```

2. Run the service

```bash
python serving_rag.py
```

3. Test the service

```bash
curl -X POST "http://localhost:8000/rag" -H "Content-Type: application/json" -d '{"query": "Which animals can hover in the air?"}'
```

## Step 2:

1. Create a new script (bash or python) to test the service with different request rates. A reference implementation is [TraceStorm](https://github.com/ServerlessLLM/TraceStorm)

## Step 3:

1. Implement a request queue to handle concurrent requests

A potential design:
Create a request queue
Put incoming requests into the queue, instead of directly processing them
Start a background thread that listens on the request queue

2. Implement a batch processing mechanism

Take up to MAX_BATCH_SIZE requests from the queue or wait until MAX_WAITING_TIME
Process the batched requests


3. Measure the performance of each step compared to the original service

4. Draw a conclusion

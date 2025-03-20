# Task 2

A FastAPI-based Retrieval-Augmented Generation (RAG) service that combines document retrieval with text generation.

> [!NOTE]  
> [serving_rag.py](serving_rag.py) is the original RAG launcher, thereby commented is found an alternative retrieve_top_k function based on the our_knn, developed in [task 1](../task-1/task.py).
> 
> [serving_rag_new.py](serving_rag_new.py) is modified to implement the request queue feature as described in Step 3 below.
> 
> [test_rag_load.py](test_rag_load.py) benchmarks a concurrently launched RAG service's performance at different request rates.
> 
> All the experiments' outputs are saved in the [Outputs](Outputs) directory, instructions to reproduce them are in the following TIP boxes.

## Step 1:

> [!TIP]
> To install and activate the environment and packages set as used during experientation, run the following:
> ```bash
> conda create -n task2env python=3.10.16 -y 
> conda activate task2env
> pip install -r requirements.txt
> ```
> Or, by extracting the original environment directly:
> ```bash
> conda env create -f task2env.yaml
> conda activate task2env
> ```

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

**Note:**  
If you encounter issues while downloading model checkpoints on a GPU machine, try the following workaround:  

1. Manually download the model on the host machine:  

```bash
conda activate rag
huggingface-cli download <model_name>
```

## Step 2:

> [!TIP]
> To run the request-rates test script (after setting the desired output length and documents of interest in [serving_rag.py](serving_rag.py), and activating the service as above):
> ```bash
> python test_rag_load.py
> ```
> To save the output in a text file too:
> ```bash
> python -u test_rag_load.py | tee output.txt
> ```
> To run the test script based on a RAG instance launched in the same prompt, use all the following commands together (suppressing the log messages and waiting about 8min for the service to activate):
> ```bash
> python serving_rag.py > /dev/null 2>&1 &
> sleep 500
> python -u test_rag_load.py | tee output.txt
> ```

Create a new script (bash or python) to test the service with different request rates. A reference implementation is [TraceStorm](https://github.com/ServerlessLLM/TraceStorm)

## Step 3:

> [!TIP]
> To activate the RAG with the request queue feature (after setting the desired output length, documents of interest, and batch-division variables in [serving_rag_new.py](serving_rag_new.py)):
> ```bash
> python serving_rag_new.py
> ```
> To run the test script based on a queue-enriched RAG instance launched in the same prompt, use all the following commands together (suppressing the log messages and waiting about 8min for the service to activate):
> ```bash
> python serving_rag_new.py > /dev/null 2>&1 &
> sleep 500
> python -u test_rag_load.py | tee output.txt
> ```

1. Implement a request queue to handle concurrent requests

    - Create a request queue
    - Put incoming requests into the queue, instead of directly processing them
    - Start a background thread that listens on the request queue

2. Implement a batch processing mechanism

    - Take up to MAX_BATCH_SIZE requests from the queue or wait until MAX_WAITING_TIME
    - Process the batched requests

3. Measure the performance of each step compared to the original service

4. Draw a conclusion
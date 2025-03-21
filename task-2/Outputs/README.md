# Task 2 - Outputs

> [!NOTE]  
> [baseline.txt](baseline.txt) contains the outcomes of testing on the original RAG service version, based on the **Qwen2.5-1.5B-Instruct** model run on **NVIDIA TITAN X Pascal** (as in all the following cases, if not differently specified).
> 
> [cpu-run.txt](cpu-run.txt) contains the same results, but with the original RAG service being run on a DICE machine's **CPU**.
> 
> [our_knn-based.txt](our_knn-based.txt) has been produced running the model with a retireval based on the *our_knn* function, defined in [task 1](../task-1/task.py).
> 
> [q-e_4-2.txt](q-e_4-2.txt), [q-e_8-4.txt](q-e_8-4.txt), and [q-e_16-4.txt](q-e_16-4.txt) contain the outcomes of testing the RAG service augmented with a batching-queue implementation. 
> 
> 
> 
> [light.txt](light.txt), [light_q-e_4-2.txt](light_q-e_4-2.txt), [light_q-e_8-4.txt](light_q-e_8-4.txt), and [light_q-e_16-4.txt](light_q-e_16-4.txt) contain...
, based on the lighter **facebook/opt-125m** model run on **NVIDIA GeForce GTX 1060 6GB**.
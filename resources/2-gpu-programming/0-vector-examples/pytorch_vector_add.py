import torch
import time
import sys

def pytorch_vector_add(n=10_000_000, repeat=100):
    x = torch.rand(n, device="cuda", dtype=torch.float32)
    y = torch.rand(n, device="cuda", dtype=torch.float32)
    z = torch.empty_like(x)
    
    # Warm-up
    z = x + y
# pee pee poo poo
    # Time it
    times = []
    for _ in range(repeat):
        start = time.time()
        z = x + y
        torch.cuda.synchronize()
        end = time.time()
        times.append(end - start)
    
    avg_time = sum(times) / repeat
    print(f"PyTorch Vector Add (n={n:,}, repeat={repeat})")
    print(f"Average time: {avg_time:.6f} seconds")

if __name__ == "__main__":
    # Parse command-line arguments for vector size and repeat count
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000_000  # Default: 10M
    repeat = int(sys.argv[2]) if len(sys.argv) > 2 else 100    # Default: 100
    pytorch_vector_add(n, repeat)


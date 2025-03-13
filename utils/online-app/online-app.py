import sys
import torch

def allocate_gpu_memory(size_in_gb):
    size_in_bytes = int(size_in_gb * 1024 * 1024 * 1024)  # Convert GB to Bytes

    try:
        # Allocate memory on GPU by creating a large tensor
        tensor = torch.empty(size_in_bytes // 4, dtype=torch.float32, device="cuda")
        print(f"Allocated {size_in_gb}GB of GPU memory. Press Enter to release...")

        input()  # Wait for user input

        # Free memory
        del tensor
        torch.cuda.empty_cache()
        print("GPU memory released.")
    except RuntimeError as e:
        print(f"Failed to allocate GPU memory: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python allocate_gpu_memory.py <size_in_gb>")
        sys.exit(1)

    try:
        size_in_gb = float(sys.argv[1])  # Read size from command line
        if size_in_gb <= 0:
            raise ValueError
        allocate_gpu_memory(size_in_gb)
    except ValueError:
        print("Invalid input. Please enter a positive number for memory size.")
        sys.exit(1)

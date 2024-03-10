singularity exec --nv torch_quantum.sif \
    python3 -m cProfile examples/mnist/mnist.py --epochs=1
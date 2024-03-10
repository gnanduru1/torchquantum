singularity exec --nv torch_quantum.sif \
    python3 examples/mnist/mnist.py --epochs 1 --wires-per-block 1
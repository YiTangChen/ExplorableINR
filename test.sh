#!bin/sh

stdbuf -e0 -o0 python3.9 -u test.py --root /path/to/data/ --dir-weights /path/to/weights/ --dir-outputs /path/to/outputs/ --batch-size 262144 --sp-sr 1.0 --sf-sr 0.10 --log-every 1 --check-every 2 --start-epoch 40 --loss MSE --dim3d 64 --dim2d 256 --dim1d 10 --spatial-fdim 64 --param-fdim 16 > test.out
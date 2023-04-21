# Note: you cannot simply `bash ...` run this in the terminal. Copy-paste each line!

# CPU
conda create -n pbc_bench_cpu python=3.9 -y
conda activate pbc_bench_cpu
conda install -c conda-forge hoomd -y
pip install jax
pip install -r requirements.txt
cd rust-neighborlist && pip install . && cd ..

# GPU
conda create -n pbc_bench_gpu python=3.9 -y
conda activate pbc_bench_gpu
conda install -c conda-forge "hoomd=*=*gpu*" -y
pip install -r requirements.txt
cd rust-neighborlist && pip install . && cd ..

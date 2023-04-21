# Benchmarking neighbors list algorithms with PBC

## Requirements

- For the rust part, on Ubuntu I first had to install `cargo`.

```bash
sudo apt install cargo
```

- To get the latest version of the `rust-neighborlist` do

```bash
pip install maturin
git clone https://github.com/mariogeiger/rust-neighborlist.git
cd rust-neighborlist
pip install .
```

Or for older versions (currently only 0.1.1 is available, which does not support PBC!) just run:

```bash
pip install rust-neighborlist
```

Or all at once with:
```bash
python3.9 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

## Findings

- `matscipy` doesn't support self-interactions, so we leave them out for this comparison.

- In 3D to have:
    - 6(+1) neighbors requires r_cutoff= dx + EPS
    - 18(+1) neighbors requires r_cutoff=sqrt(2) dx + EPS
    - 26(+1) neighbors requires r_cutoff=sqrt(3) dx + EPS
-> we selected the 26 neighbors case.

- 
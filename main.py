import numpy as np
# import jax.numpy as jnp
# import ase
# from ase.neighborlist import neighbor_list as ase_nl
# from neighborlist import neighbor_list as rust_nl
from matscipy.neighbours import neighbour_list as matscipy_nl
# from jax_md import space, partition

import matplotlib.pyplot as plt
import time


def pos_init_cartesian_3d(box, dx):
    box_vector = np.ones((3,)) * box
    n = np.array((box_vector / dx).round(), dtype=int)
    grid = np.meshgrid(range(n[0]), range(n[1]), range(n[2]), indexing='xy')
    r = (np.vstack(list(map(np.ravel, grid))).T + 0.5) * dx
    return r

def sanity_check_comparison():
    box = 1.0


    cell = np.eye(3) * box
    pbc = np.array([True, True, True])

        
    dx = 0.25

    pos = pos_init_cartesian_3d(box, dx)
    cutoff = dx * 3**(0.5) + 1e-8

    # ase
    a = ase.Atoms(positions=pos, pbc=pbc)
    res1 = ase_nl("ij", a=a, cutoff=cutoff)

    # matscipy
    res2 = matscipy_nl("ij", cutoff=cutoff, positions=pos, cell=np.eye(3), pbc=pbc)

    # rust
    *res3, _, _, _ = rust_nl(
        positions=pos, cutoff=cutoff, cell=np.eye(3), pbc=pbc, self_interaction=False)

    # jax-md
    displacement_fn, _ = space.periodic(1.0)
    jaxmd_nl_fn = partition.neighbor_list(
        displacement_fn, 
        box=box, 
        r_cutoff=cutoff, 
        format=partition.NeighborListFormat.Sparse, 
        capacity_multiplier=1.0)
    jaxmd_nl = jaxmd_nl_fn.allocate(pos)
    jaxmd_nl = jaxmd_nl.update(pos)

    ij1 = np.array(res1) # (2, num_edges)
    ij2 = np.array(res2)
    ij3 = np.array(res3)
    ij4 = np.array(jaxmd_nl.idx)

    # check if the same edges are contained in each result
    assert (ij1.T[:, None] == ij2.T).all(axis=2).any(axis=0).all()
    assert (ij1.T[:, None] == ij3.T).all(axis=2).any(axis=0).all()
    assert (ij1.T[:, None] == ij4.T).all(axis=2).any(axis=0).all()

def timing_comparison():
    # timing comparison
    N = 2
    box = 1.0
    cell = np.eye(3) * box
    pbc = np.array([True, True, True])

    dxs = [0.5, 0.25, 0.2, 0.15, 0.1, 0.05, 0.025, 0.02, 0.015, 0.01] #, 0.008]
    poss = [pos_init_cartesian_3d(box, dx) for dx in dxs]
    cutoffs = [dx * 3**(0.5) + 1e-8 for dx in dxs]
    atoms = [ase.Atoms(positions=pos, pbc=pbc) for pos in poss]
    
    times_list = []

    # ase
    temp = []
    for i, dx in enumerate(dxs[:3]):
        start = time.time()
        for _ in range(N):
             res1 = ase_nl("ij", a=atoms[i], cutoff=cutoffs[i])
        print(res1[0].shape)
        temp.append((time.time()-start)/N)
    times_list.append(temp)

    # matscipy
    temp = []
    for i, dx in enumerate(dxs):
        start = time.time()
        for _ in range(N):
            res2 = matscipy_nl(
                "ij", cutoff=cutoffs[i], positions=poss[i], cell=cell, pbc=pbc)
        print(res2[0].shape)
        temp.append((time.time()-start)/N)
    times_list.append(temp)
    
    # rust
    temp = []
    for i, dx in enumerate(dxs):
        start = time.time()
        for _ in range(N):
            *res3, _, _, _ = rust_nl(
                positions=poss[i], 
                cutoff=cutoffs[i], 
                cell=cell, 
                pbc=pbc, 
                self_interaction=False
            )
        print(res3[0].shape)
        temp.append((time.time()-start)/N)
    times_list.append(temp)
        
    # jax-md
    temp = []
    for i, dx in enumerate(dxs[:9]):
        displacement_fn, shift_fn = space.periodic(1.0)
        jaxmd_nl_fn = partition.neighbor_list(
            displacement_fn, 
            box=box, 
            r_cutoff=cutoffs[i], 
            format=partition.NeighborListFormat.Sparse, 
            capacity_multiplier=1.0
        )
        jaxmd_nl = jaxmd_nl_fn.allocate(poss[i])
        jaxmd_nl = jaxmd_nl.update(poss[i])
        jaxmd_nl.idx.block_until_ready()
        pos_i = jnp.array(poss[i])
        
        start = time.time()
        for _ in range(N):
            jaxmd_nl = jaxmd_nl.update(pos_i)
            jaxmd_nl.idx.block_until_ready()
        print(jaxmd_nl.idx[0].shape)
        temp.append((time.time()-start)/N)
    times_list.append(temp)
    
    labels = ["ase", "matscipy", "rust", "jax-md"]
    nx = np.round(box/np.array(dxs))
    plt.figure()
    for i, time_i in enumerate(times_list):
        plt.plot(nx[:len(time_i)], time_i, label=labels[i])
    plt.xlabel("number of particles per dimension")
    plt.ylabel("neighbors search time [s]")
    plt.title("""Timing of neighbors search algorithms on a 3D periodic box
              (Cartesian grid positions; time averaged over 2 runs)""")
    plt.legend()
    plt.grid()
    plt.savefig("timing_comparison.png")


def test_matscipy():
    
    # timing comparison
    N = 2
    box = 1.0
    cell = np.eye(3) * box
    pbc = np.array([True, True, True])

    dxs = [0.5, 0.25, 0.2, 0.15, 0.1, 0.05, 0.025, 0.02, 0.015, 0.01, 0.00793, 0.00585]
    poss = [pos_init_cartesian_3d(box, dx) for dx in dxs]
    cutoffs = [dx * 3**(0.5) + 1e-8 for dx in dxs]
    
    # matscipy
    times_list = []
    for i, dx in enumerate(dxs):
        start = time.time()
        for _ in range(N):
            res2 = matscipy_nl(
                "ij", cutoff=cutoffs[i], positions=poss[i], cell=cell, pbc=pbc)
        print(res2[0].shape)
        times_list.append((time.time()-start)/N)
    
    nx = np.round(box/np.array(dxs)) ** 3
    plt.figure()
    plt.plot(nx, times_list, 'o-', label="matscipy")
    plt.xlabel("number of particles")
    plt.ylabel("neighbors search time [s]")
    plt.title("Matscipy")
    plt.legend()
    plt.grid()
    plt.savefig("matscipy_large.png")

    
if __name__ == "__main__":
    
    # sanity_check_comparison()
    # timing_comparison()
    test_matscipy()
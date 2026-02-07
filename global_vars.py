import cnn
import numpy as np
import sys
import jax.numpy as jnp
import math
import os
from itertools import product, permutations
from netket.graph import Lattice
from matplotlib import pyplot as plt

L = None

def coor2in(x, y, L):
    return int(((x+L)%L)*(L) + (y+(L))%(L))

def in2coor(i,L):
    y = i % L
    x = (i-y) // L 
    return [x, y]

def inverse(perm):
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return jnp.array(inv)

def compose(p1, p2):
    """Return composition of two permutations: p1 ◦ p2"""
    return jnp.array([p2[i] for i in p1])

def construct_product_table(G):
    """Construct table A(i, j) such that G[A[i][j]] =  G[i]^{-1}*G[j]"""
    n = len(G)
    A = [[-1 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        g_inv = inverse(G[i])
        for j in range(n):
            target = compose(g_inv, G[j])
            for k in range(n):
                if np.array_equal(G[k], target):
                    A[i][j] = k
                    break
    return jnp.array(A)


def generate_mask(N_cell, ker_size, L):

    n_elms = ker_size**2
    ker = np.full((N_cell, N_cell), n_elms)
    
    n = 0
    for del_i, del_j in product(range(ker_size), range(ker_size)):
        for i, j in product(range(L), range(L)):
            c1 = coor2in(i, j, L)
            i2, j2 = i + del_i, j + del_j
            c2 = coor2in(i2, j2, L)

            ker[c1, c2] = n

        n += 1

    return jnp.array(ker)

def update_globals():
    global N_plaquette, N, sstart, kernel2, kernel3, permutation, translation_site, translation_cell, inverse_permutation
    global kernel2, kernel3, symmetries, point_group, inverse_translation, graph, product_table, point_group_table
    global kx, ky, triangle_list

    if L is not None:

        sqrt3 = np.sqrt(3.0)
        basis = np.array([[1.0, 0.0], [0.5, sqrt3 / 2.0],])
        cell = np.array([basis[0] / 2.0, basis[1] / 2.0, (basis[0]+basis[1])/2.0])
        dimensions = [L, L]
        graph = Lattice(basis_vectors=basis, site_offsets=cell, extent=dimensions, pbc=True)
        # graph.draw()
        # plt.show()

        N_plaquette = L**2
        N = graph.n_nodes

        symmetries = graph.automorphisms()
        print(len(symmetries))

        permutation = symmetries.to_array()
        target = np.array([2, 3, 4, 5, 6, 3*L, 3*L+1, 3*L+2, 3*L+3, 3*L+4, 3*L+5, 6*L+1])
        # target = np.array([3, 5, 7, 8, 9, 10])
        mask = np.apply_along_axis(lambda row: np.array_equal(np.sort(row[target]), target), axis=1, arr=permutation)
        point_group = jnp.array(permutation[mask])
        
        translation_site = jnp.array(graph.translation_group().to_array())
        translation_cell = jnp.array([ s[3*j].item() // 3 for s in translation_site for j in range(N_plaquette) ]).reshape(N_plaquette, N_plaquette)

        print("point group: ", len(point_group)," translation_group: ",len(translation_site))

        # point_group_table = construct_product_table(point_group)
        # translation_group_table = construct_product_table(translation_cell)
        
        permutation = jnp.array([compose(translation_site[j], point_group[i]) for j in range(N_plaquette) for i in range(len(point_group)) ])
        # inverse_permutation = jnp.array([inverse(s) for s in permutation])
        # inverse_translation = jnp.array([inverse(s) for s in translation_site])
        # print(inverse_permutation.shape)


        # if not os.path.exists(f"product_table_L{L}.npy"):

        #     A = np.zeros((len(symmetries), len(symmetries)), int)
        #     for it, jt in product(range(N_plaquette), range(N_plaquette)):    
        #         k = translation_group_table[it, jt]
                
        #         for ip, jp in product(range(len(point_group)), range(len(point_group))):
        #             index = [ip + it * len(point_group), jp + jt * len(point_group)]
                    
        #             T = compose(inverse(point_group[ip]), compose(translation_site[k], point_group[ip]))
        #             translated_index = T.tolist().index(0) // 3
        #             A[index[0], index[1]] = int(point_group_table[ip, jp] + translated_index * len(point_group))

        #     np.save(f"product_table_L{L}.npy", A.astype(int))
        #     A = jnp.array(A).astype(int)
        #     exit()

        # else:
        #     A = np.load(f"product_table_L{L}.npy")
        
        # product_table = jnp.array([ [A[A[i, 0], A[j, 0]] for j in range(len(symmetries)) ] for i in range(len(symmetries)) ])
        sstart = np.zeros((N, ), dtype = int)
        
        n = 0
        for i, j in product(range(L), range(L)):
            sstart[3*n + 1], sstart[3*n + 2] = 1, -1
            sstart[3*n + 0] = 1 if i % 2 == 0 else -1

            n += 1

        sstart = jnp.array(sstart)

        kernel2 = generate_mask(N_plaquette, L//2, L)
        kernel3 = generate_mask(N_plaquette, L, L)

        kx, ky = [], []
        for i, j in product(range(L), range(L)):
            kx.extend([np.exp(2j*np.pi*(i+cell[0, 0])/L),  np.exp(2j*np.pi*(i+cell[1, 0])/L), np.exp(2j*np.pi*(i+cell[2, 0])/L)])
            ky.extend([np.exp(2j*np.pi*(j+cell[0, 1])/L),  np.exp(2j*np.pi*(j+cell[1, 1])/L), np.exp(2j*np.pi*(j+cell[2, 1])/L)])

        kx, ky = jnp.array(kx), jnp.array(ky)

        triangle_list = [[3*coor2in(i,j,L), 3*coor2in(i,j,L)+1, 3*coor2in(i,j,L)+2] for i in range(L) for j in range(L)]
        triangle_list.extend([[3*coor2in(i,j,L), 3*coor2in(i,j-1,L)+2, 3*coor2in(i+1,j-1,L)+1] for i in range(L) for j in range(L)])
        triangle_list = jnp.array(triangle_list).astype(int)
import netket as nk
import numpy as np
import jax.numpy as jnp

L = None


def coor2in(x, y, L):
    return int(((x + L) % L) * L + ((y + L) % L))


def generate_mask(n_cell, ker_size, L):
    n_elms = ker_size**2
    ker = np.full((n_cell, n_cell), n_elms)

    n = 0
    for del_i in range(ker_size):
        for del_j in range(ker_size):
            for i in range(L):
                for j in range(L):
                    c1 = coor2in(i, j, L)
                    c2 = coor2in(i + del_i, j + del_j, L)
                    ker[c1, c2] = n
            n += 1

    return jnp.array(ker)


def _subgroup_table(parent_table, subgroup_parent_indices):
    parent_to_sub = {parent_idx: sub_idx for sub_idx, parent_idx in enumerate(subgroup_parent_indices)}
    subgroup_table = np.vectorize(parent_to_sub.get)(parent_table[np.ix_(subgroup_parent_indices, subgroup_parent_indices)])
    return jnp.array(subgroup_table)


def update_globals():
    global N_plaquette, N
    global graph, basis_vectors, site_offsets, site_positions, reciprocal_basis
    global translation_group, translation_site, translation_table, translation_cell
    global translation_even_site, translation_odd_site
    global point_group_obj, point_group, point_group_labels, point_group_indices, point_group_table
    global full_group, full_group_array, symmetries
    global little_group_obj, little_group, little_group_labels, little_group_readable_labels
    global little_group_indices, little_group_parent_indices, little_group_table
    global kM, kx, ky, sstart, sstart_a1, kernel2, kernel3, form_factor_M
    global translation_a1_idx, translation_a2_idx

    if L is None:
        return

    graph = nk.graph.Kagome([L, L], pbc=True)

    N_plaquette = L**2
    N = graph.n_nodes

    basis_vectors = np.array(graph.basis_vectors)
    site_offsets = np.array(graph.site_offsets)
    site_positions = jnp.array(np.array(graph.positions))

    reciprocal_basis = 2 * np.pi * np.linalg.inv(basis_vectors).T
    kM = np.array([np.pi, np.pi / np.sqrt(3.0)])

    translation_group = graph.translation_group()
    translation_site = jnp.array(translation_group.to_array())
    translation_table = jnp.array(translation_group.product_table)
    translation_cell = jnp.array(
        [[perm[3 * j].item() // 3 for j in range(N_plaquette)] for perm in translation_site]
    )

    point_group_obj = graph.point_group()
    point_group = jnp.array(point_group_obj.to_array())
    point_group_labels = [repr(elem) for elem in point_group_obj.elems]
    point_group_indices = {label: i for i, label in enumerate(point_group_labels)}
    point_group_table = jnp.array(point_group_obj.product_table)

    full_group = graph.space_group()
    full_group_array = jnp.array(full_group.to_array())
    symmetries = full_group

    little_group_obj = full_group.little_group(*kM)
    little_group_readable_labels = [repr(elem) for elem in little_group_obj.elems]
    little_group_parent_indices = [point_group_indices[label] for label in little_group_readable_labels]
    little_group = point_group[jnp.array(little_group_parent_indices)]
    little_group_labels = ["e", "R3", "tauR", "Rtau"]
    little_group_indices = {label: i for i, label in enumerate(little_group_labels)}
    little_group_table = _subgroup_table(np.array(point_group_table), little_group_parent_indices)

    translation_a1_idx, translation_a2_idx = None, None
    for idx in range(len(translation_site)):
        translated_cell = int(translation_cell[idx, 0])
        if translated_cell == coor2in(1, 0, L):
            translation_a1_idx = idx
        elif translated_cell == coor2in(0, 1, L):
            translation_a2_idx = idx

    translation_x = np.array(translation_cell[:, 0]) // L
    translation_even_site = translation_site[jnp.array(np.where(translation_x % 2 == 0)[0])]
    translation_odd_site = translation_site[jnp.array(np.where(translation_x % 2 == 1)[0])]

    sstart = np.zeros((N,), dtype=int)
    n = 0
    for i in range(L):
        for j in range(L):
            sstart[3 * n + 1], sstart[3 * n + 2] = 1, -1
            sstart[3 * n + 0] = 1 if i % 2 == 0 else -1
            n += 1
    sstart = jnp.array(sstart)
    sstart_a1 = sstart[translation_site[translation_a1_idx]]

    kernel2 = generate_mask(N_plaquette, L // 2, L)
    kernel3 = generate_mask(N_plaquette, L, L)

    kx, ky = [], []
    for i in range(L):
        for j in range(L):
            for offset in site_offsets:
                kx.append(np.exp(2j * np.pi * (i + offset[0]) / L))
                ky.append(np.exp(2j * np.pi * (j + offset[1]) / L))
    kx, ky = jnp.array(kx), jnp.array(ky)

    form_factor_M = jnp.exp(1j * (site_positions @ jnp.array(kM)))

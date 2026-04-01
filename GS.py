# Kagome spin liquid
import os
from pyexpat import features
from re import L
import sys
from traceback import print_list
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import jax
os.environ["JAX_PLATFORM_NAME"] = "gpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "platform"
os.environ["nk.config.netket_experimental_disable_ode_jit"] = "True"
os.environ["NETKET_EXPERIMENTAL_FFT_AUTOCORRELATION"] = "True"

from copy import copy
import json
import time
import math
import netket as nk
import optax
import netket.experimental as nkx
import numpy as np
import random
import shutil
import functools

from netket.operator.spin import sigmax, sigmaz, sigmay, identity
import jax.numpy as jnp
import scipy
import jax.tree_util
from matplotlib import pyplot as plt
import pickle 

import flax
from flax import struct
import flax.linen as nn
import netket.nn as nknn
from flax import traverse_util
from flax.core import freeze
import qutip as qtp
from tqdm import tqdm
import time

from typing import Any, Optional, Tuple
from netket.utils.types import PyTree, PRNGKeyT
from netket.sampler import MetropolisRule
from netket.utils import HashableArray, mpi
from netket.stats import statistics as mpi_statistics
from functools import partial
import utils
import global_vars as g
import gc

from netket.graph import Lattice

import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process some global variables.')
    parser.add_argument('--L', type=int, required=True, help='Value for L')
    parser.add_argument('--n_features', type=int, required=True, help='Feature numbers')
    parser.add_argument('--mode', type=str, required=True, help='Feature numbers')

    args = parser.parse_args()
    g.L = args.L
    g.update_globals()
    return args

from collections import defaultdict

def get_scalar_indices_by_top_key(pytree):
    result = {}
    counter = 0

    for key, subtree in pytree.items():
        indices = []

        def count_and_collect(leaf):
            nonlocal counter
            # Flatten each array leaf to scalars
            n = leaf.size
            idxs = list(range(counter, counter + n))
            counter += n
            indices.extend(idxs)

        jax.tree_util.tree_map(count_and_collect, subtree)
        result[key] = indices

    return result

def main():
    
    args = parse_arguments()
    print(args, "num_spin = ", g.N)

    @nk.hilbert.random.random_state.dispatch
    def random_state(hilb : nk.hilbert.Spin, key, size : int, *, dtype):
        out = jnp.stack([g.sstart] * size, axis = 0)
        swap = jax.random.randint(key, shape=(size,), minval=0, maxval = 6 * g.N_plaquette)
        edges = jnp.array(g.graph.edges())

        @jax.vmap
        def flip(sigma, key):
            index_swap = edges[key]
            index2 = jnp.array([index_swap[1], index_swap[0]])
            s = sigma.at[index_swap].set(sigma[index2])
            return s.astype(dtype)
        
        outp = flip(out, swap)
        return outp


    hi = nk.hilbert.Spin(s = 1/2, N=g.N, total_sz=0, inverted_ordering = True)
    ha = nk.operator.Heisenberg(hilbert=hi, J = 1/4, graph=g.graph, sign_rule=False)

    exchange_rule = nk.sampler.rules.ExchangeRule(clusters = g.graph.edges())
    sampler = nk.sampler.MetropolisSampler(hi, exchange_rule, sweep_size = g.N // 5, n_chains=2**8, reset_chains = False)
    
    #model = utils.CNN_exchange(n_features = args.n_features, Jastrow = True, deep_CNN = False)
    #### Vanilla complex-RBM wavefunction
    model = nk.models.RBM(alpha = 16, dtype = complex)
    #### Vanilla CNN (GCNN with only translational symmetry)
    model = nk.models.GCNN(symmetries=g.translation_group, layers = 4, features=(6,) * 4, parity = 1,
                param_dtype = complex, characters=g.symmetries.character_table()[0])(x)
    vstate = nk.vqs.MCState(sampler, model = model, n_samples=2**13, chunk_size=2**14, n_discard_per_chain=10)
    # param_map = get_scalar_indices_by_top_key(vstate.parameters)
    
    print(vstate.n_parameters)
    E = utils.evolve2(vstate, ha, 200, 2, 1e-6, show_progress=True) 

    return()
    Jastrow_params = flax.core.copy(vstate.parameters["Jastrow_exchange_0"])
    model = utils.CNN_exchange(n_features = args.n_features)
    
    vstate = nk.vqs.MCState(sampler, model = model, n_samples=2**12, chunk_size=2**14, n_discard_per_chain=10)
    params = flax.core.unfreeze(vstate.parameters)
    params["Jastrow_exchange_0"] = Jastrow_params
    vstate.parameters = jax.tree_util.tree_map(lambda x: x, params)

    print(vstate.n_parameters)
    E = utils.evolve2(vstate, ha, 400, 4, 1e-6, show_progress=True) 

if __name__ == "__main__":
    main()

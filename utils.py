import math
import netket as nk
import numpy as np
import random
import shutil
import functools
import pickle, json

from netket.operator.spin import sigmax, sigmaz, sigmay, identity
import jax.numpy as jnp
from scipy import sparse
import jax.tree_util

import flax
from flax import struct, nnx
import flax.linen as nn
import netket.nn as nknn
from flax import traverse_util
from flax.core import freeze

import global_vars as g
from netket import jax as nkjax
from netket.utils import HashableArray, mpi
from netket.stats import Stats, statistics as mpi_statistics
from netket.optimizer.qgt.qgt_jacobian_dense import convert_tree_to_dense_format
from netket.jax import tree_cast, logsumexp_cplx
import netket.experimental as nkx
import gc
from tqdm import tqdm
from functools import partial
from scipy.stats import gmean
from matplotlib import pyplot as plt
from jax.nn.initializers import lecun_normal, zeros
from netket.nn.activation import reim_selu as selu
    
def XYZ(hi, site1, site2):
    D = np.array([[1, 0, 0, 0],[0, -1, 0.5, 0],[0, 0.5, -1, 0],[0, 0, 0, 1]])
    D = sparse.coo_matrix(D)
    return  nk.operator.LocalOperator(hi, D, [site1, site2], dtype=float).to_jax_operator()
    
@jax.jit
def activation2(x):
    return x/2 + x**2/4

@jax.jit
def activation4(x):
    return x/2 + x**2/4 - x**4/48

def AF_Hamiltonian(hi):

    ha = 0 * identity(hi)

    down_triangle_list = [[3*g.coor2in(i,j,g.L), 3*g.coor2in(i,j,g.L)+1, 3*g.coor2in(i,j,g.L)+2] for i in range(g.L) for j in range(g.L)]
    up_triangle_list = [[3*g.coor2in(i,j,g.L), 3*g.coor2in(i,j-1,g.L)+2, 3*g.coor2in(i+1,j-1,g.L)+1] for i in range(g.L) for j in range(g.L)]

    for u in up_triangle_list:
        ha += XYZ(hi, u[0], u[1]) + XYZ(hi, u[1], u[2]) + XYZ(hi, u[0], u[2]) 

    for u in down_triangle_list:
        ha += XYZ(hi, u[0], u[1]) + XYZ(hi, u[1], u[2]) + XYZ(hi, u[0], u[2]) 

    return ha

    
class conv2(nn.Module):

    out_features: int
    bias: bool = True
    ker_size: int = 2
    dtype: type =  complex
    W_scale: float = 1.0

    @nn.compact
    def __call__(self,x):
        
        size = x.shape
        N_kernel = self.ker_size**2

        W_t = jnp.pad(self.param('W_t',nn.initializers.normal(stddev = 0.5 / np.sqrt(N_kernel * size[-1])), (N_kernel, self.out_features, size[-1]), self.dtype), 
                        pad_width = ((0, 1), (0, 0), (0, 0)), constant_values = 0j)
        if self.ker_size == g.L // 2:
            kernel = jnp.take(W_t, g.kernel2, axis = 0)
        else:
            kernel = jnp.take(W_t, g.kernel3, axis = 0) * jnp.where(g.kernel2 == (g.L//2)**2, 0.1, 1)[:, :, None, None]
        
        y = jax.lax.dot_general(x, kernel, (( (1, 2), (1, 3)), ((), ())))

        if self.bias:
            b =  self.param('b',zeros, (self.out_features,), self.dtype)        
            y += b[None, None, :]
        
        return y
    

def exchange(x0):
        
    s2 = (1 + x0) // 2
    x_shift = jnp.round(jnp.angle(jnp.sum(g.kx[None, :] * s2, axis = -1)) * g.L/(2*np.pi), 5)
    x_shift = jnp.where(x_shift <= 0, g.L - jnp.ceil(-x_shift),  -jnp.ceil(-x_shift)).astype(int) % g.L

    a = jnp.sum(jnp.abs(x0 - g.sstart[None, :]), axis = 1)*2 + (x_shift % 2)
    b = jnp.sum(jnp.abs(x0 - g.sstart[None, g.translation_site[g.L]]), axis = 1)*2 + (1 - x_shift % 2)
    
    mask = jnp.stack([a < b] * x0.shape[1], axis = 1)
    x = jnp.where(mask, x0 - g.sstart[None, :] - 0.1, x0 - g.sstart[None, g.translation_site[g.L]] - 0.1)
    x = x.reshape(-1, g.N_plaquette, 3)

    return x

class GCNN(nn.Module):

    @nn.compact
    def __call__(self, x):
        
        n_batch = x.shape[0]
            
        x0 = x[:, g.point_group].reshape(-1, g.N_plaquette, 3)
        mean_field = jnp.sum(x0 * conv2(out_features = 3, ker_size = g.L, bias = False)(x0), axis = (1, 2))
        mean_field = logsumexp_cplx(mean_field.reshape(n_batch, -1), axis=[1])
        
        out0 = mean_field + nk.models.GCNN(symmetries=g.symmetries, layers = 4, features=(6,) * 4, parity = 1,
                param_dtype = complex, characters=g.symmetries.character_table()[0])(x)
        
        return out0

class Jastrow_exchange(nn.Module):

    @nn.compact
    def __call__(self, pair):
        mean_field = jnp.sum(pair * conv2(out_features = 3, ker_size = g.L)(pair), axis = (1, 2))
        return mean_field
    
class deep_CNN(nn.Module):
    n_features: int

    @nn.compact
    def __call__(self, pair, x):
        u = jnp.concatenate((pair, x.reshape(-1, g.N_plaquette, 3)), axis = -1)
        u = activation2(conv2(out_features = self.n_features, ker_size = g.L)(u))
        u = activation2(conv2(out_features = self.n_features, ker_size = g.L//2)(u))
        u = activation4(conv2(out_features = self.n_features, ker_size = g.L//2)(u))

        return jnp.sum(u, axis = [1, 2]) / jnp.sqrt(self.n_features)

class CNN_exchange(nn.Module):
    
    n_features: int
    Jastrow: bool = True
    deep_CNN: bool = True

    @nn.compact
    def __call__(self, x0):
        
        n_batch = x0.shape[0]
        x = x0[:, g.point_group].reshape(-1, g.N)
        x = jnp.concatenate((x, -x), axis = 1).reshape(-1, g.N)

        pair = exchange(x)
        out = Jastrow_exchange()(pair) if self.Jastrow else jnp.zeros((x.shape[0]), dtype = complex)

        if self.deep_CNN:
            out +=  deep_CNN(self.n_features)(pair, x)
        
        out0 = logsumexp_cplx(out.reshape(n_batch, -1), axis=[1])

        return out0


def evolve(vstate, h0, nstep, dt, rcond, show_progress = False): 

    def single_update(vstate):  

        E_loc = vstate.local_estimators(h0)
        E = mpi_statistics(E_loc)
        E_loc = E_loc.reshape(-1)
        ΔE_loc = (E_loc - E.mean)

        s  = vstate.samples.reshape(-1, N)
        O = nkjax.jacobian(vstate._apply_fun, vstate.parameters, s, vstate.model_state,
                mode="complex", dense=True, center=True)[:, :, prange]

        O = (O - jnp.mean(O, axis = 0)[None, :]) / (np.sqrt(vstate.n_samples))
        O = O[:, 0, :] + 1j*O[:, 1, :]
        Sd = O.conj().T @ O  + rcond * jnp.eye(O.shape[1])
        OEdata = O.conj() * (ΔE_loc[:, None] / (np.sqrt(vstate.n_samples)))
        F = jnp.sum(OEdata, axis=0)

        ev, V = jnp.linalg.eigh(jnp.real(Sd))
        rho = V.conj().T @ F

        # #### Regularizing based on rotated F / eigs of S
    
        ev_inv = rho/ev
        filter = jnp.where(jnp.abs(ev/ev[-1]) > 1e-5, 1e1, jnp.where(jnp.abs(ev/ev[-1]) > 1e-8, 0.1, 0.01 ))
        ev_inv = jnp.where((jnp.abs(ev_inv) >= filter), filter * ev_inv/jnp.abs(ev_inv), ev_inv)

        update = jnp.zeros(2 * vstate.n_parameters, float)
        update = update.at[prange].set(jnp.real(V @ ev_inv))

        y, reassemble = convert_tree_to_dense_format(vstate.parameters, "complex")
        dw = tree_cast (reassemble(update), vstate.parameters)

        return dw, E

    N = h0.hilbert.size
    lr = -dt/nstep
    loop = tqdm(range(nstep)) if show_progress else range(nstep)

    n = 0
    for i in loop:

        if n % 50 == 0:
            param_range = np.sort(np.random.choice(vstate.n_parameters, (4000,), replace = False)) if vstate.n_parameters > 4000 else np.arange(vstate.n_parameters)
            prange = jnp.array(param_range) if param_range is not None else jnp.arange(vstate.n_parameters)
            prange = jnp.concatenate((prange, vstate.n_parameters + prange), axis = 0)
        
        old_pars = vstate.parameters
        k1, E = single_update(vstate)
        vstate.parameters = jax.tree_util.tree_map(lambda x, y: x + lr*y , old_pars, k1)
        k2, E = single_update(vstate)
        vstate.parameters = jax.tree_util.tree_map(lambda x, y1, y2: x + 0.5*lr*(y1+y2) , old_pars, k1, k2)
        
        if show_progress:
            loop.set_description(str(E))

        n += 1
    
    return E

def evolve2(vstate, h0, nstep, dt, rcond, show_progress = True): 


    def single_update(vstate, E_loc, param_range):  

        n_samples = len(E_loc)
        n_params = len(param_range)
        param_select = jnp.concatenate((param_range, param_range + vstate.n_parameters), axis = 0)

        E = mpi_statistics(E_loc)
        ΔE_loc = (E_loc - E.mean) / np.sqrt(n_samples)


        jacobians = nkjax.jacobian(vstate._apply_fun, vstate.parameters, vstate.samples.reshape(-1, g.N), vstate.model_state,
            mode="complex", dense=True, center=True, chunk_size=vstate.chunk_size,)[:, :, param_select]
        jacobians = jacobians / np.sqrt(n_samples)
        jacobians = jacobians[:, 0, :] + 1j * jacobians[:, 1, :]

        Sd = mpi.mpi_sum_jax(jacobians.conj().T @ jacobians)[0] + rcond*jnp.eye(jacobians.shape[1])
        OEdata = jacobians.conj() * ΔE_loc[:, None]
        F = nk.stats.sum(OEdata, axis=0)
        ################

        if temperature > 1e-6:
            phi_loc = vstate.log_value(vstate.samples.reshape((-1, h0.hilbert.size)))
            Δphi_loc = (phi_loc - jnp.mean(phi_loc))/np.sqrt(n_samples)

            OSdata = jacobians.conj() * Δphi_loc[:, None]
            F += temperature * nk.stats.sum(OSdata, axis=0)

        #### Regularizing based on rotated F / eigs of S
        ev, V = jnp.linalg.eigh(jnp.real(Sd))
        rho = V.T @ jnp.real(F)

        #### Regularizing based on rotated F / eigs of S
    
        ev_inv = rho/ev
        filter = jnp.where(jnp.abs(ev/ev[-1]) > 1e-5, 1e1, jnp.where(jnp.abs(ev/ev[-1]) > 1e-8, 0.5, 0.01 ))
        ev_inv = V @ jnp.where((jnp.abs(ev_inv) >= filter), filter * ev_inv/jnp.abs(ev_inv), ev_inv)
        
        update = jnp.zeros((vstate.n_parameters), dtype = complex)
        update = update.at[param_range].set(ev_inv[:n_params] + 1j * ev_inv[n_params:])

        y, reassemble = convert_tree_to_dense_format(vstate.parameters, "holomorphic")
        dw = tree_cast (reassemble(update), vstate.parameters)

        return dw

    lr = -dt/nstep
    n = 0
    loop = tqdm(range(nstep)) if show_progress else range(nstep)


    E_last = vstate.local_estimators(h0).reshape(-1)
    for i in loop:
        
        temperature = 2.0 if n <= 100 else 2*np.exp(-0.3*(n-100))
        param_range = np.sort(np.random.choice(vstate.n_parameters, (4000,), replace = False)) if vstate.n_parameters > 4000 else np.arange(vstate.n_parameters)

        old_pars = vstate.parameters
        k1 = single_update(vstate, E_last,  param_range)
        
        vstate.parameters = jax.tree_util.tree_map(lambda x, y: x + lr*y , old_pars, k1)
        E_loc = vstate.local_estimators(h0).reshape(-1)

        k2 = single_update(vstate, E_loc, param_range)
        vstate.parameters = jax.tree_util.tree_map(lambda x, y1, y2: x + 0.5*lr*(y1+y2) , old_pars, k1, k2)
        E_loc = vstate.local_estimators(h0).reshape(-1)

        E_last = E_loc

        n += 1
        if show_progress:
            loop.set_description(str(mpi_statistics(E_last)) + str(vstate.sampler_state.acceptance))

    return mpi_statistics(E_last)
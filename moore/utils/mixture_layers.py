# Copyright 2021 The PODNN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=====================================================================================
'''
Implementation is adapted from https://github.com/caisr-hh/podnn.
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize # new
from copy import deepcopy

n_models_global = 5
agg_out_dim = 3


class InputLayer(nn.Module):

    """
       InputLayer stucture the data in a parallel form ready to be consumed by
       the upcoming parallel layes.
    """


    def __init__(self,n_models):

        """
        Arg:
            n_models: number of individual models within the ensemble
        """

        super(InputLayer, self).__init__()
        self.n_models = n_models
        global n_models_global
        n_models_global = self.n_models

    def forward(self,x):

        """
        Arg
            x: is the input to the network as in other standard deep neural networks.

        return:
            x_parallel: is the parallel form of the received input (x).
        """

        x_parallel = torch.unsqueeze(x,0)
        x_parallel_next = torch.unsqueeze(x, 0)
        for i in range(1,self.n_models):
            x_parallel =  torch.cat((x_parallel,x_parallel_next),axis=0)

        return x_parallel


class ParallelLayer(nn.Module):

    """
        Parallellayer creates a parallel layer from the structure of unit_model it receives.
    """

    def __init__(self, unit_model):

        """
        Arg:
            unit_model: specifies what computational module each unit of the parallel layer contains.
                        unit_model is a number of layer definitions followed by each other.
        """

        super(ParallelLayer,self).__init__()
        self.n_models = n_models_global
        self.model_layers = []
        for i in range(self.n_models):
            for j in range(len(unit_model)):
                try:
                    unit_model[j].reset_parameters()
                except:
                    pass
            self.model_layers.append(deepcopy(unit_model))
        self.model_layers = nn.ModuleList(self.model_layers)

    def forward(self, x):

        """
        Arg:
            x: is the parallel input with shape [n_models,n_samples,dim] for fully connected layers.

        return:
            parallel_output: is the output formed by applying modules within each units on the input.

            shape: [n_models,n_samples,dim] for fully connected layers..

        """

        parallel_output = self.model_layers[0](x[0])
        parallel_output = torch.unsqueeze(parallel_output,0)
        for i in range(1,self.n_models):
            next_layer = self.model_layers[i](x[i])
            next_layer = torch.unsqueeze(next_layer, 0)
            parallel_output = torch.cat((parallel_output,next_layer),0)

        return parallel_output

def compute_angles(basis):
    cos = torch.abs(torch.clip(basis@torch.transpose(basis, 1, 2), -1, 1))
    rad = torch.arccos(cos)
    deg = torch.rad2deg(rad)

    off_diagonal_angles = deg[:, ~torch.eye(deg.shape[-1], dtype=bool)]

    assert torch.all(torch.isclose(torch.min(off_diagonal_angles), torch.tensor(90.))), torch.min(off_diagonal_angles)

class OrthogonalLayer1D(nn.Module):

    """
        OrthogonalLayer1D make the outputs of each unit of the previous sub-layer orthogonal to each other.
        Orthogonalization is performed using Gram-Schmidt orthogonalization.
    """

    def __init__(self):
        super(OrthogonalLayer1D, self).__init__()
        self.projection_matrix = None

    def forward(self,x):

        """
        Arg:
            x: The parallel formated input with shape: [n_models,n_samples,dim]

        return:
            basis: Orthogonalized version of the input (x). The shape of basis is [n_models,n_samples,dim].
                   For each sample, the outputs of all of the models (n_models) will be orthogonal
                   to each other.
        """
        # initialise matrix
        if self.projection_matrix is None:
            self.projection_matrix = torch.nn.Parameter(torch.eye(x.shape[2], x.shape[2]))

        return project_to_unique_subspaces(x, self.projection_matrix)


        x1 = torch.transpose(x, 0,1)
        basis = torch.unsqueeze(x1[:, 0, :] / (torch.unsqueeze(torch.linalg.norm(x1[:, 0, :], axis=1), 1)), 1)

        for i in range(1, x1.shape[1]):
            v = x1[:, i, :]
            v = torch.unsqueeze(v, 1)
            w = v - torch.matmul(torch.matmul(v, torch.transpose(basis, 2, 1)), basis)
            wnorm = w / (torch.unsqueeze(torch.linalg.norm(w, axis=2), 2))
            basis = torch.cat([basis, wnorm], axis=1)

        basis = torch.transpose(basis,0,1)
        return basis


def project_to_unique_subspaces(
    U: torch.Tensor,
    A: torch.Tensor
) -> torch.Tensor:
    """
    Args:
      U: (batch, K, dim)                — MoE outputs
      A: (dim, dim)                     — unconstrained parameter
    Returns:
      V: (batch, K, dim)                — each expert in its own orthogonal subspace
    """
    batch, K, dim = U.shape
    base, rem = divmod(dim, K)      # e.g. for dim=100, K=6 → base=16, rem=4
    # first `rem` experts get (base+1) dims, the rest get base dims
    sizes = [(base + 1) if i < rem else base for i in range(K)]
    starts = [0] + list(torch.cumsum(torch.tensor(sizes), 0).tolist())

    # build Cayley Q as before
    S = A - A.t()
    I = torch.eye(dim, device=A.device, dtype=A.dtype)
    Q = torch.linalg.solve(I - S, I + S)  # (dim, dim)

    V = torch.zeros_like(U)
    for i in range(K):
        s, e = starts[i], starts[i+1]
        Bi = Q[:, s:e]           # shape (dim, sizes[i])
        ui = U[:, i]             # shape (batch, dim)
        coords = ui @ Bi         # → (batch, sizes[i])
        V[:, i] = coords @ Bi.t()# → (batch, dim)
    return V
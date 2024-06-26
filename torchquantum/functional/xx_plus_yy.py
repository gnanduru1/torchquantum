import functools
import torch
import numpy as np

from typing import Callable, Union, Optional, List, Dict, TYPE_CHECKING
from ..macro import C_DTYPE, F_DTYPE, ABC, ABC_ARRAY, INV_SQRT2
from ..util.utils import pauli_eigs, diag
from torchpack.utils.logging import logger
from torchquantum.util import normalize_statevector

from .gate_wrapper import gate_wrapper, apply_unitary_einsum, apply_unitary_bmm

if TYPE_CHECKING:
    from torchquantum.device import QuantumDevice
else:
    QuantumDevice = None


def xxplusyy_matrix(params):
    """Compute unitary matrix for XXplusYY gate.

    Args:
        params (torch.Tensor): The rotation angle. (Theta,Beta)

    Returns:
        torch.Tensor: The computed unitary matrix.

    """
    theta = params[:, 0].unsqueeze(dim=-1).type(C_DTYPE)
    beta = params[:, 1].unsqueeze(dim=-1).type(C_DTYPE)

    co = torch.cos(theta / 2)
    si = torch.sin(theta / 2)

    return torch.stack(
        [
            torch.cat(
                [
                    torch.tensor([[1]]),
                    torch.tensor([[0]]),
                    torch.tensor([[0]]),
                    torch.tensor([[0]]),
                ],
                dim=-1,
            ),
            torch.cat(
                [
                    torch.tensor([[0]]),
                    co,
                    (-1j * si * torch.exp(1j * beta)),
                    torch.tensor([[0]]),
                ],
                dim=-1,
            ),
            torch.cat(
                [
                    torch.tensor([[0]]),
                    (-1j * si * torch.exp(-1j * beta)),
                    co,
                    torch.tensor([[0]]),
                ],
                dim=-1,
            ),
            torch.cat(
                [
                    torch.tensor([[0]]),
                    torch.tensor([[0]]),
                    torch.tensor([[0]]),
                    torch.tensor([[1]]),
                ],
                dim=-1,
            ),
        ],
        dim=-2,
    ).squeeze(0)


_xxplusyy_mat_dict = {
    "xxplusyy": xxplusyy_matrix,
}


def xxplusyy(
    q_device,
    wires,
    params=None,
    n_wires=None,
    static=False,
    parent_graph=None,
    inverse=False,
    comp_method="bmm",
):
    """Perform the XXPlusYY gate.

    Args:
        q_device (tq.QuantumDevice): The QuantumDevice.
        wires (Union[List[int], int]): Which qubit(s) to apply the gate.
        params (torch.Tensor, optional): Parameters (if any) of the gate.
            Default to None.
        n_wires (int, optional): Number of qubits the gate is applied to.
            Default to None.
        static (bool, optional): Whether use static mode computation.
            Default to False.
        parent_graph (tq.QuantumGraph, optional): Parent QuantumGraph of
            current operation. Default to None.
        inverse (bool, optional): Whether inverse the gate. Default to False.
        comp_method (bool, optional): Use 'bmm' or 'einsum' method to perform
        matrix vector multiplication. Default to 'bmm'.

    Returns:
        None.

    """
    name = "xxplusyy"
    mat = _xxplusyy_mat_dict[name]
    gate_wrapper(
        name=name,
        mat=mat,
        method=comp_method,
        q_device=q_device,
        wires=wires,
        params=params,
        n_wires=n_wires,
        static=static,
        parent_graph=parent_graph,
        inverse=inverse,
    )

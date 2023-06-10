from torchquantum.algorithms import Hamiltonian
import numpy as np

def test_hamiltonian():
    coeffs = [1.0, 1.0]
    paulis = ["ZZ", "ZX"]
    hamil = Hamiltonian(coeffs, paulis)
    assert np.allclose(
        hamil.get_matrix().cpu().detach().numpy(), 
        np.array(
            [[ 1.+0.j,  1.+0.j,  0.+0.j,  0.+0.j],
            [ 1.+0.j, -1.+0.j,  0.+0.j,  0.+0.j],
            [ 0.+0.j,  0.+0.j, -1.+0.j, -1.+0.j],
            [ 0.+0.j,  0.+0.j, -1.+0.j,  1.+0.j]]))
    

    coeffs = [0.6]
    paulis = ["XXZ"]
    hamil = Hamiltonian(coeffs, paulis)
    assert np.allclose(
        hamil.get_matrix().cpu().detach().numpy(), 
        np.array(
            [[ 0.0000+0.j,  0.0000+0.j,  0.0000+0.j,  0.0000+0.j,  0.0000+0.j,  0.0000+0.j,
          0.6000+0.j,  0.0000+0.j],
        [ 0.0000+0.j, -0.0000+0.j,  0.0000+0.j, -0.0000+0.j,  0.0000+0.j, -0.0000+0.j,
          0.0000+0.j, -0.6000+0.j],
        [ 0.0000+0.j,  0.0000+0.j,  0.0000+0.j,  0.0000+0.j,  0.6000+0.j,  0.0000+0.j,
          0.0000+0.j,  0.0000+0.j],
        [ 0.0000+0.j, -0.0000+0.j,  0.0000+0.j, -0.0000+0.j,  0.0000+0.j, -0.6000+0.j,
          0.0000+0.j, -0.0000+0.j],
        [ 0.0000+0.j,  0.0000+0.j,  0.6000+0.j,  0.0000+0.j,  0.0000+0.j,  0.0000+0.j,
          0.0000+0.j,  0.0000+0.j],
        [ 0.0000+0.j, -0.0000+0.j,  0.0000+0.j, -0.6000+0.j,  0.0000+0.j, -0.0000+0.j,
          0.0000+0.j, -0.0000+0.j],
        [ 0.6000+0.j,  0.0000+0.j,  0.0000+0.j,  0.0000+0.j,  0.0000+0.j,  0.0000+0.j,
          0.0000+0.j,  0.0000+0.j],
        [ 0.0000+0.j, -0.6000+0.j,  0.0000+0.j, -0.0000+0.j,  0.0000+0.j, -0.0000+0.j,
          0.0000+0.j, -0.0000+0.j]]))
    
    hamil = Hamiltonian.from_file("test/algorithms/h2.txt")

    assert np.allclose(
        hamil.matrix.cpu().detach().numpy(), 
        np.array(
            [[-1.0636533 +0.j,  0.        +0.j,  0.        +0.j,
         0.        +0.j,  0.        +0.j,  0.        +0.j,
         0.1809312 +0.j,  0.        +0.j],
       [ 0.        +0.j, -1.0636533 +0.j,  0.        +0.j,
         0.        +0.j,  0.        +0.j,  0.        +0.j,
         0.        +0.j,  0.1809312 +0.j],
       [ 0.        +0.j,  0.        +0.j, -1.8369681 +0.j,
         0.        +0.j,  0.1809312 +0.j,  0.        +0.j,
         0.        +0.j,  0.        +0.j],
       [ 0.        +0.j,  0.        +0.j,  0.        +0.j,
        -1.8369681 +0.j,  0.        +0.j,  0.1809312 +0.j,
         0.        +0.j,  0.        +0.j],
       [ 0.        +0.j,  0.        +0.j,  0.1809312 +0.j,
         0.        +0.j, -0.24521835+0.j,  0.        +0.j,
         0.        +0.j,  0.        +0.j],
       [ 0.        +0.j,  0.        +0.j,  0.        +0.j,
         0.1809312 +0.j,  0.        +0.j, -0.24521835+0.j,
         0.        +0.j,  0.        +0.j],
       [ 0.1809312 +0.j,  0.        +0.j,  0.        +0.j,
         0.        +0.j,  0.        +0.j,  0.        +0.j,
        -1.0636533 +0.j,  0.        +0.j],
       [ 0.        +0.j,  0.1809312 +0.j,  0.        +0.j,
         0.        +0.j,  0.        +0.j,  0.        +0.j,
         0.        +0.j, -1.0636533 +0.j]]))
    print("hamiltonian test passed!")

if __name__ == '__main__':
    import pdb
    pdb.set_trace()
    test_hamiltonian()
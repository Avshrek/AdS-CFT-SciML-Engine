import numpy as np
import os
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from tqdm import tqdm

# --- PRO CONFIGURATION ---
NUM_SAMPLES = 10000    # The "Brute Force" amount needed for <1% error
GRID_SIZE = 64         # Standard FNO resolution
SAVE_DIR = 'data_holography'
# -------------------------

def construct_laplace_matrix(n):
    """
    Constructs the exact finite-difference matrix for the Laplace equation.
    Solves for the entire universe at once using Linear Algebra.
    """
    N = n * n
    # Diagonals
    main_diag = -4 * np.ones(N)
    side_diag = np.ones(N - 1)
    side_diag[np.arange(1, N) % n == 0] = 0  # Remove wrap-around connections
    up_down_diag = np.ones(N - n)
    
    # Construct Sparse Matrix
    diagonals = [main_diag, side_diag, side_diag, up_down_diag, up_down_diag]
    offsets = [0, -1, 1, -n, n]
    A = sp.diags(diagonals, offsets, shape=(N, N), format='lil')
    
    # Enforce Boundary Conditions (Dirichlet)
    # The Top Row (indices 0 to n-1) is the Input.
    # The Left, Right, and Bottom are fixed to 0.
    
    # Identify boundary indices
    grid_indices = np.arange(N).reshape(n, n)
    top_row = grid_indices[0, :]
    bottom_row = grid_indices[-1, :]
    left_col = grid_indices[:, 0]
    right_col = grid_indices[:, -1]
    
    all_boundaries = np.unique(np.concatenate((top_row, bottom_row, left_col, right_col)))
    
    # Set rows corresponding to boundaries to Identity (1*x = value)
    for idx in all_boundaries:
        A[idx, :] = 0
        A[idx, idx] = 1
        
    return A.tocsc(), all_boundaries, top_row

def generate_exact_physics():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    print(f"🚀 Building Sparse Physics Matrix ({GRID_SIZE}x{GRID_SIZE})...")
    A, all_boundaries, top_row_indices = construct_laplace_matrix(GRID_SIZE)
    
    # Pre-factorize the matrix (This makes solving instant)
    print("🧠 Pre-calculating Inverse Matrix (LU Factorization)...")
    solve_fast = spla.factorized(A)
    
    input_data = []
    output_data = []
    
    print(f"⚡ Generating {NUM_SAMPLES} universes with Exact Physics...")
    for _ in tqdm(range(NUM_SAMPLES)):
        # 1. Create a Random Quantum Boundary (The Input)
        x = np.linspace(0, 2*np.pi, GRID_SIZE)
        
        # Complex wave composition to force the AI to learn
        k1 = np.random.randint(1, 5)
        k2 = np.random.randint(5, 12)
        phase = np.random.uniform(0, 2*np.pi)
        
        boundary_val = np.sin(k1 * x + phase) + 0.5 * np.cos(k2 * x) 
        boundary_val = boundary_val * np.random.uniform(0.8, 1.2) # Amplitude scaling
        
        # 2. Setup the Right-Hand Side (b vector)
        b = np.zeros(GRID_SIZE * GRID_SIZE)
        
        # Apply boundary conditions to b
        # Top row gets our random wave
        b[top_row_indices] = boundary_val
        # Other boundaries stay 0 (Vacuum)
        
        # 3. SOLVE Ax = b (The Magic Step)
        flattened_universe = solve_fast(b)
        universe = flattened_universe.reshape(GRID_SIZE, GRID_SIZE)
        
        input_data.append(boundary_val)
        output_data.append(universe)
        
    print("💾 Saving Datasets...")
    np.save(f'{SAVE_DIR}/boundary_train.npy', np.array(input_data))
    np.save(f'{SAVE_DIR}/bulk_train.npy', np.array(output_data))
    print("✅ DONE. Exact physics data ready.")

if __name__ == "__main__":
    generate_exact_physics()

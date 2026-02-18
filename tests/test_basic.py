import numpy as np
import pytest
import scipy.sparse as sp
from linalg_lib import SprsMat

# ==========================================
# Fixtures (Setup Data)
# ==========================================


@pytest.fixture
def simple_diagonal_data():
    """
    Creates data for a 3x3 Diagonal Matrix:
    [[10,  0,  0],
     [ 0, 20,  0],
     [ 0,  0, 30]]
    """
    row_ptr = [0, 1, 2, 3]
    col_indx = [0, 1, 2]
    vals = [10.0, 20.0, 30.0]
    shape = (3, 3)
    return row_ptr, col_indx, vals, shape


# ==========================================
# Tests
# ==========================================


def test_manual_construction(simple_diagonal_data):
    """Test that we can pass raw lists to the constructor."""
    row_ptr, col_indx, vals, shape = simple_diagonal_data

    # 1. Initialize
    mat = SprsMat(row_ptr, col_indx, vals, shape)

    # 2. Check Shape
    assert mat.shape() == shape

    # 3. Check Vals
    assert mat.vals() == vals


def test_zeros_static_method():
    """Test the static `zeros` constructor."""
    rows, cols = 5, 10

    # 1. Call static method
    mat = SprsMat.zeros(rows, cols)

    # 2. Verify structure
    assert mat.shape() == (rows, cols)
    assert mat.vals() == []  # Should have no stored values

    # 3. Verify access returns 0.0
    assert mat[(0, 0)] == 0.0
    assert mat[(4, 9)] == 0.0


def test_getitem_access(simple_diagonal_data):
    """Test accessing elements via mat[(i, j)]."""
    row_ptr, col_indx, vals, shape = simple_diagonal_data
    mat = SprsMat(row_ptr, col_indx, vals, shape)

    # 1. Check Non-Zeros (The diagonal)
    assert mat[(0, 0)] == 10.0
    assert mat[(1, 1)] == 20.0
    assert mat[(2, 2)] == 30.0

    # 2. Check Zeros (Implicit)
    assert mat[(0, 1)] == 0.0
    assert mat[(1, 0)] == 0.0
    assert mat[(2, 0)] == 0.0


def test_vs_scipy_random():
    """
    'Oracle Test': Generate a random matrix in Scipy, feed its internal
    CSR arrays to our Rust struct, and ensure behavior is identical.
    """
    # 1. Generate random sparse matrix via Scipy
    # density=0.2 means 20% of items are non-zero
    scipy_mat = sp.random(10, 10, density=0.2, format="csr", dtype=np.float64)

    # 2. Extract internal CSR buffers
    # .tolist() is used because your current constructor expects Vec<T>, not numpy arrays
    row_ptr = scipy_mat.indptr.tolist()
    col_indices = scipy_mat.indices.tolist()
    data = scipy_mat.data.tolist()
    shape = scipy_mat.shape

    # 3. Build Rust Matrix
    rust_mat = SprsMat(row_ptr, col_indices, data, shape)

    # 4. Compare Shape
    assert rust_mat.shape() == shape

    # 5. Compare 50 Random Point Accesses
    for _ in range(50):
        r = np.random.randint(0, shape[0])
        c = np.random.randint(0, shape[1])

        scipy_val = scipy_mat[r, c]  # Truth
        rust_val = rust_mat[(r, c)]  # Your Impl

        assert rust_val == scipy_val, f"Mismatch at ({r}, {c})"


def test_out_of_bounds_check():
    """
    Ensure your Rust code doesn't panic (crash) on bad indices.
    (Note: You might need to add logic in Rust to return a Result/Error
    if you haven't yet, otherwise this test expects a panic or garbage data).
    """
    mat = SprsMat.zeros(5, 5)

    # Depending on your Rust impl, this might panic or raise PyIndexError
    # If you implemented error handling:
    with pytest.raises(IndexError):
        _ = mat[(100, 100)]

import numpy as np
import pytest
import scipy.sparse as sp
from pytest import approx
from linalg_lib import SprsMat

# ==========================================
# 1. Structural Edge Cases
# ==========================================


def test_weird_shapes():
    """
    Test 0x0, 0xN, and Nx0 matrices.
    These often cause crashes due to empty vectors or 'index out of bounds' on row_ptr[0].
    """
    # 0x0 Matrix
    mat = SprsMat.zeros(0, 0)
    assert mat.shape() == (0, 0)
    assert mat.vals() == []
    with pytest.raises(IndexError):
        _ = mat[(0, 0)]

    # 5x0 Matrix (5 rows, 0 columns) -> Should have row_ptr=[0,0,0,0,0,0]
    mat = SprsMat.zeros(5, 0)
    assert mat.shape() == (5, 0)
    assert len(mat.vals()) == 0
    with pytest.raises(IndexError):
        _ = mat[(0, 0)]

    # 0x5 Matrix (0 rows, 5 columns) -> Should have row_ptr=[0]
    mat = SprsMat.zeros(0, 5)
    assert mat.shape() == (0, 5)
    with pytest.raises(IndexError):
        _ = mat[(0, 0)]


def test_boundary_access():
    """
    Test the EXACT corners of the matrix.
    This catches 'off-by-one' errors in your bounds checking or row_ptr logic.
    """
    rows, cols = 10, 10
    mat = SprsMat.zeros(rows, cols)

    # We manually inject values into the corners for testing
    # Note: Since your current API constructs from vectors, we'll build a custom one
    # Matrix with points at (0,0), (0,9), (9,0), (9,9)
    row_ptr = [0, 2, 2, 2, 2, 2, 2, 2, 2, 3, 4]  # Skip middle rows
    col_indx = [0, 9, 0, 9]  # Row 0 has (0,9), Row 9 has (0,9)
    # Wait, the logic above is tricky to write manually. Let's use scipy to help generate inputs.

    coo = sp.coo_matrix(([1, 2, 3, 4], ([0, 0, 9, 9], [0, 9, 0, 9])), shape=(10, 10))
    csr = coo.tocsr()

    rust_mat = SprsMat(
        csr.indptr.tolist(), csr.indices.tolist(), csr.data.tolist(), csr.shape
    )

    # 1. Top-Left
    assert rust_mat[(0, 0)] == 1.0
    # 2. Top-Right
    assert rust_mat[(0, 9)] == 2.0
    # 3. Bottom-Left
    assert rust_mat[(9, 0)] == 3.0
    # 4. Bottom-Right
    assert rust_mat[(9, 9)] == 4.0

    # 5. Out of bounds (Just outside)
    with pytest.raises(IndexError):
        _ = rust_mat[(10, 0)]
    with pytest.raises(IndexError):
        _ = rust_mat[(0, 10)]
    with pytest.raises(IndexError):
        _ = rust_mat[
            (-1, 0)
        ]  # Python handles negative indices, but your Rust might not expect them yet!


# ==========================================
# 2. Logic & Precision Tests
# ==========================================


def test_explicit_zeros():
    """
    Does the library distinguish between 'missing' zero and 'stored' zero?
    Some sparse algorithms break if you explicitly store a 0.0 value.
    """
    # Matrix: [[0.0, 5.0]]
    # We explicitly store the 0.0 at (0,0)
    row_ptr = [0, 2]
    col_indx = [0, 1]
    vals = [0.0, 5.0]
    shape = (1, 2)

    mat = SprsMat(row_ptr, col_indx, vals, shape)

    # Should return the stored 0.0, not a default 0.0
    # (Functionally the same value, but verifies storage logic)
    assert mat[(0, 0)] == 0.0
    assert mat[(0, 1)] == 5.0
    assert len(mat.vals()) == 2  # verify it didn't strip the zero


def test_floating_point_special_values():
    """
    Test NaNs and Infinities.
    Rust f64 supports them, but index matching logic shouldn't break.
    """
    row_ptr = [0, 2]
    col_indx = [0, 1]
    vals = [float("nan"), float("inf")]
    shape = (1, 2)

    mat = SprsMat(row_ptr, col_indx, vals, shape)

    val_nan = mat[(0, 0)]
    val_inf = mat[(0, 1)]

    assert np.isnan(val_nan)
    assert np.isinf(val_inf)


# ==========================================
# 3. The "Fuzzer" (Stress Test)
# ==========================================


@pytest.mark.parametrize("iteration", range(20))
def test_fuzz_compare_scipy(iteration):
    """
    Generates 20 random matrices with:
    - Random dimensions (1x1 to 100x100)
    - Random densities (empty to full)
    - Checks 100 random points per matrix
    """
    # 1. Random Parameters
    rows = np.random.randint(1, 50)
    cols = np.random.randint(1, 50)
    density = np.random.uniform(0, 1.0)  # 0% to 100% dense

    # 2. Build Oracle (Scipy)
    sp_mat = sp.random(rows, cols, density=density, format="csr", dtype=np.float64)

    # 3. Build Candidate (Rust)
    rust_mat = SprsMat(
        sp_mat.indptr.tolist(),
        sp_mat.indices.tolist(),
        sp_mat.data.tolist(),
        sp_mat.shape,
    )

    # 4. Verify Metadata
    assert rust_mat.shape() == sp_mat.shape
    assert len(rust_mat.vals()) == sp_mat.nnz

    # 5. Verify Random Access (The Stress Test)
    # Check 100 random coordinates
    for _ in range(100):
        r = np.random.randint(0, rows)
        c = np.random.randint(0, cols)

        expected = sp_mat[r, c]
        actual = rust_mat[(r, c)]

        # Use approx to ignore float diffs
        assert actual == approx(expected), (
            f"Failed iter {iteration}: Mismatch at ({r},{c}) in {rows}x{cols} mat"
        )


def test_full_matrix_access():
    """
    Test a matrix that is 100% dense.
    This ensures logic doesn't rely on 'skips' to work.
    """
    # 3x3 Full matrix
    dense = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    sp_mat = sp.csr_matrix(dense)

    mat = SprsMat(
        sp_mat.indptr.tolist(),
        sp_mat.indices.tolist(),
        sp_mat.data.tolist(),
        sp_mat.shape,
    )

    for r in range(3):
        for c in range(3):
            assert mat[(r, c)] == dense[r, c]

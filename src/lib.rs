mod sparse_mat;

use sparse_mat::*;

use pyo3::prelude::*;

#[pymodule]
fn linalg_lib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SprsMat>()?;
    Ok(())
}

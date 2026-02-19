use pyo3::{exceptions::PyIndexError, prelude::*};

#[pyclass]
#[derive(Default)]
pub struct SprsMat {
    pub row_ptr: Vec<usize>,
    pub col_indx: Vec<usize>,
    vals: Vec<f64>,
    #[pyo3(get)]
    pub shape: (usize, usize),
}

#[pymethods]
impl SprsMat {
    #[new]
    pub fn new(
        row_ptr: Vec<usize>,
        col_indx: Vec<usize>,
        vals: Vec<f64>,
        shape: (usize, usize),
    ) -> Self {
        SprsMat {
            row_ptr,
            col_indx,
            vals,
            shape,
        }
    }

    #[staticmethod]
    pub fn zeros(m: usize, n: usize) -> Self {
        SprsMat {
            shape: (m, n),
            row_ptr: vec![0; m + 1],
            ..Default::default()
        }
    }

    pub fn vals(&self) -> Vec<f64> {
        self.vals.clone()
    }

    // fn shape(&self) -> (usize, usize) {
    //     self.shape
    // }

    fn __getitem__(&self, idx: [i32; 2]) -> PyResult<f64> {
        let [m, n] = idx;
        let m =
            usize::try_from(m).map_err(|_| PyIndexError::new_err("index m cannot be negative"))?;
        let n =
            usize::try_from(n).map_err(|_| PyIndexError::new_err("index n cannot be negative"))?;

        if self.shape.0 <= m || self.shape.1 <= n {
            return Err(PyIndexError::new_err(format!(
                "Index {:?} out of bounds for  matrix of shape {:?}",
                idx, self.shape
            )));
        };
        Ok(
            match self.col_indx[self.row_ptr[m]..self.row_ptr[m + 1]]
                .iter()
                .position(|x| x == &n)
            {
                Some(i) => self.vals[self.row_ptr[m] + i],
                None => 0.0,
            },
        )
    }

    fn __setitem__(&mut self, idx: [i32; 2], value: f64) -> PyResult<()> {
        let [m, n] = idx;
        let m =
            usize::try_from(m).map_err(|_| PyIndexError::new_err("index m cannot be negative"))?;
        let n =
            usize::try_from(n).map_err(|_| PyIndexError::new_err("index n cannot be negative"))?;

        if self.shape.0 <= m || self.shape.1 <= n {
            return Err(PyIndexError::new_err(format!(
                "Index {:?} out of bounds for  matrix of shape {:?}",
                idx, self.shape
            )));
        };

        match self.col_indx[self.row_ptr[m]..self.row_ptr[m + 1]]
            .iter()
            .rposition(|x| x <= &n)
        {
            Some(rel_indx) => {
                if self.col_indx[self.row_ptr[m] + rel_indx] == n {
                    self.vals[self.row_ptr[m] + rel_indx] = value;
                } else {
                    self.vals.insert(self.row_ptr[m] + rel_indx, value);
                    self.row_ptr.insert(self.row_ptr[m] + rel_indx, n);
                    self.col_indx.iter_mut().for_each(|x| *x += 1);
                }
            }
            None => {
                todo!()
            }
        };

        Ok(())
    }
}

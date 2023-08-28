use std::iter::zip;

#[derive(Debug)]
pub struct Gcm {
    x: Vec<f64>,
    mu: Vec<f64>,
}
impl Gcm {
    pub fn interpolate(&self, z: f64) -> f64 {
        match self.x.binary_search_by(|x_j| x_j.partial_cmp(&z).unwrap()) {
            Ok(j) => {
                // An exact match on a binary search is inherently safe.
                unsafe { self.mu.get_unchecked(j).clone() }
            }
            Err(j) => {
                // We must determine where to interpolate from.
                let k = self.x.len();
                if j == 0 {
                    self.mu[1]
                        + (self.mu[0] - self.mu[1]) / (self.x[0] - self.x[1]) * (z - self.x[1])
                    // self.mu[0]
                    //     + (self.mu[0] - self.mu[1]) / (self.x[0] - self.x[1]) * (z - self.x[0])
                } else if j == k {
                    self.mu[k - 2]
                        + (self.mu[k - 1] - self.mu[k - 2]) / (self.x[k - 1] - self.x[k - 2])
                            * (z - self.x[k - 2])
                } else {
                    // z < x[j] => z - x[j] < 0
                    self.mu[j - 1]
                        + (self.mu[j] - self.mu[j - 1]) / (self.x[j] - self.x[j - 1])
                            * (z - self.x[j - 1])
                    // self.mu[j]
                    //     + (self.mu[j - 1] - self.mu[j]) / (self.x[j - 1] - self.x[j])
                    //         * (z - self.x[j])
                }
            }
        }
    }
    pub fn x<'a>(&'a self) -> &'a Vec<f64> {
        &self.x
    }
    pub fn mu<'a>(&'a self) -> &'a Vec<f64> {
        &self.mu
    }
}

/// Construct the greatest convex minorant of the sequence of points
/// *(xᵢ, f(xᵢ)), i = 0,...,n-1*, assuming that (1)
/// (1) the values satisfy xᵢ < xᵢ₊₁ for i = 0,...,n-2,
/// (2) -inf < xᵢ < inf ∀i, and
/// (3) xᵢ is not NaN ∀i.
/// The result of the algorithm is essentially meaningless if these
/// three conditions are not satisfied. The implementation
/// will not panic, and instead return a result which should be regarded
/// as meaningless.
/// In addition to the three conditions above, at `n` must be at least 2,
/// and `x.len() == y.len()` must hold. Failure to satisfy these two
/// conditions will result in a panic.
pub fn gcm(x: &[f64], y: &[f64]) -> Gcm {
    gcm_ltor(x.to_vec(), y.to_vec())
}

fn diff(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    if n > 1 {
        let mut dx: Vec<f64> = vec![0.0; n - 1];

        x.windows(2)
            .zip(dx.iter_mut())
            .for_each(|(w, dx_i)| *dx_i = w[1] - w[0]);

        dx
    } else {
        vec![]
    }
}

pub fn gcm_rtol(x: Vec<f64>, y: Vec<f64>) -> Gcm {
    // If we want to permit unsorted x values, then one must include
    // the following 3 lines. Most likely, this should be handled in a
    // separate place, as the possibility of duplicates are not dealt with.
    // let mut z: Vec<_> = x.into_iter().zip(y.into_iter()).collect();
    // z.sort_unstable_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap());
    // let (x, y): (Vec<_>, Vec<_>) = z.into_iter().unzip();

    // These two necessary conditions could be handled more delicately.
    let n = y.len();
    assert_eq!(x.len(), n);
    assert!(n > 1);

    let mut nu = diff(&y);
    let mut dx = diff(&x);
    let k = nu.len();
    let mut w: Vec<usize> = vec![1; k];
    let mut j = k - 1;
    loop {
        // let k = nu.len();
        // let mut j = k - 1;
        while j > 0 && nu[j - 1] / dx[j - 1] <= nu[j] / dx[j] {
            j -= 1;
        }
        if j == 0 {
            let mut nu_out = y;
            let mut pos: usize = 1;
            // for i in 0..nu.len() {
            //     let mu = nu[i] / dx[i];
            //     for _ in 0..w[i] {
            //         nu_out[pos] = nu_out[pos - 1] + mu * (x[pos] - x[pos - 1]);
            //         pos += 1;
            //     }
            // }
            // The maximum value of `pos` is 1 + ∑ⱼwⱼ = 1 + (n - 1) = n, but the
            // last offset accessed is n - 1. Hence, all uses of `pos` are safe.
            for (nu_i, (dx_i, w_i)) in zip(nu, zip(dx, w)) {
                let mu = nu_i / dx_i;
                for _ in 0..w_i {
                    nu_out[pos] = nu_out[pos - 1] + mu * (x[pos] - x[pos - 1]);
                    pos += 1;
                }
            }
            return Gcm { x, mu: nu_out };
        }
        let w_prime = w[j - 1] + w[j];
        let w_j_m1 = w[j - 1] as f64;
        let w_j = w[j] as f64;
        let nu_prime = (w_j_m1 * nu[j - 1] + w_j * nu[j]) / w_prime as f64;
        let dx_prime = (w_j_m1 * dx[j - 1] + w_j * dx[j]) / w_prime as f64;
        nu.remove(j);
        w.remove(j);
        dx.remove(j);
        nu[j - 1] = nu_prime;
        w[j - 1] = w_prime;
        dx[j - 1] = dx_prime;
        // Adjacent violators were pooled, thus check the newly formed block
        // against the (new) preceding block. However, if we pooled the
        // penultimate and last blocks, then no (new) preceding block exists,
        // and we must move the index left.
        j = j.min(nu.len() - 1);
    }
}

pub fn gcm_ltor(x: Vec<f64>, y: Vec<f64>) -> Gcm {
    // These two necessary conditions could be handled more delicately.
    let n = y.len();
    assert_eq!(x.len(), n);
    assert!(n > 1);

    let v = diff(&y);
    let dx = diff(&x);

    let n = v.len();
    let mut nu: Vec<f64> = Vec::with_capacity(n);
    nu.push(v[0]);
    let mut xi: Vec<f64> = Vec::with_capacity(n);
    xi.push(dx[0]);
    let mut w: Vec<usize> = Vec::with_capacity(n);
    w.push(1);
    let mut j: usize = 0;
    let mut i: usize = 1;
    while i < n {
        j += 1;
        // SAFETY: `i` is always less than `n`, hence, always a valid index.
        nu.push(v[i]);
        xi.push(dx[i]);
        w.push(1);
        // SAFETY: `j` is always the index of the last block, and when `j = 0`,
        // neither the second conditional nor loop body are executed.
        i += 1;
        while j > 0 && nu[j - 1] / xi[j - 1] > nu[j] / xi[j] {
            let w_prime = w[j - 1] + w[j];
            let nu_prime = (w[j - 1] as f64 * nu[j - 1] + w[j] as f64 * nu[j]) / w_prime as f64;
            let xi_prime = (w[j - 1] as f64 * xi[j - 1] + w[j] as f64 * xi[j]) / w_prime as f64;
            nu[j - 1] = nu_prime;
            xi[j - 1] = xi_prime;
            w[j - 1] = w_prime;
            nu.pop();
            xi.pop();
            w.pop();
            j -= 1;
        }
    }
    let mut f = y;
    let mut f_prev = f[0];
    let mut pos: usize = 1;
    for (nu_j, (xi_j, w_j)) in zip(nu, zip(xi, w)) {
        let dfdx_j = nu_j / xi_j;
        for (f_pos, dx_pos) in f[pos..pos + w_j]
            .iter_mut()
            .zip(dx[pos - 1..pos - 1 + w_j].iter())
        {
            *f_pos = f_prev + dfdx_j * *dx_pos;
            f_prev = f_pos.clone();
        }
        pos += w_j;
    }
    Gcm { x, mu: f }
}

#[derive(Debug)]
pub struct Lcm {
    g: Gcm,
}
impl Lcm {
    pub fn interpolate(&self, z: f64) -> f64 {
        self.g.interpolate(z)
    }

    pub fn x<'a>(&'a self) -> &'a Vec<f64> {
        &self.g.x
    }
    pub fn mu<'a>(&'a self) -> &'a Vec<f64> {
        &self.g.mu
    }
}
pub fn lcm(x: &[f64], y: &[f64]) -> Lcm {
    let x = x.to_vec();
    let y: Vec<f64> = y.iter().map(|y_i| -*y_i).collect();
    let mut g = gcm_ltor(x, y);
    g.mu.iter_mut().for_each(|mu_i| *mu_i = -*mu_i);
    Lcm { g }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diff_works() {
        let x: Vec<f64> = vec![4.0, 2.5, 3.5, 3.0];
        assert_eq!(diff(&x), vec![-1.5, 1.0, -0.5]);

        let x: Vec<f64> = vec![1.0, 3.0];
        assert_eq!(diff(&x), vec![2.0]);

        let x: Vec<f64> = vec![1.0];
        assert_eq!(diff(&x), vec![]);

        let x: Vec<f64> = vec![];
        assert_eq!(diff(&x), vec![]);
    }

    fn example_1() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let x: Vec<f64> = vec![1.0, 3.0, 6.0, 10.0, 11.0, 13.0, 17.0, 20.0];

        let y: Vec<f64> = vec![
            1.755940276352825,
            1.3378194316497374,
            2.6098971934850894,
            4.283002134408102,
            3.8957003420178404,
            2.3619868614176722,
            4.460741607606607,
            2.787487520958698,
        ];

        let mu_gcm: Vec<f64> = vec![
            1.755940276352825,
            1.3378194316497374,
            1.5936432121160244,
            1.934741586071074,
            2.0200161795598364,
            2.1905653665373612,
            2.531663740492411,
            2.787487520958698,
        ];
        let mu_lcm: Vec<f64> = vec![
            1.755940276352825,
            2.3175095781428867,
            3.159863530827979,
            4.283002134408102,
            4.308393487722174,
            4.359176194350319,
            4.460741607606607,
            2.787487520958698,
        ];
        (x, y, mu_gcm, mu_lcm)
    }

    fn example_2() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 7.0];
        let y: Vec<f64> = vec![1.0, 3.0, 2.0, 5.0, 6.0];
        let mu_gcm: Vec<f64> = vec![1.0, 1.5, 2.0, 3.0, 6.0];
        let mu_lcm: Vec<f64> = vec![1.0, 3.0, 4.0, 5.0, 6.0];
        (x, y, mu_gcm, mu_lcm)
    }

    #[test]
    fn gcm_example_1_works() {
        let (x, y, mu, _) = example_1();
        let g = gcm_ltor(x, y);
        assert_eq!(g.mu(), &mu);
    }
    #[test]
    fn gcm_example_1_interpolation_works() {
        let (x, y, mu, _) = example_1();
        let g = gcm_ltor(x, y);
        let z: f64 = 5.0;
        assert_eq!(g.interpolate(z), 1.508368618627262);

        let x: Vec<f64> = vec![1.0, 3.0, 6.0, 10.0, 11.0, 13.0, 17.0, 20.0];
        for (x_i, mu_i) in x.into_iter().zip(mu.into_iter()) {
            assert_eq!(g.interpolate(x_i), mu_i);
        }

        let z: f64 = -1.0;
        assert_eq!(g.interpolate(z), 2.1740611210559124);

        let z: f64 = 25.0;
        assert_eq!(g.interpolate(z), 3.21386048840251);
    }

    #[test]
    fn lcm_example_1_works() {
        let (x, y, _, mu) = example_1();
        let l = lcm(&x, &y);
        assert_eq!(l.mu(), &mu);
    }

    #[test]
    fn lcm_example_1_interpolation_works() {
        let (x, y, _, mu) = example_1();
        let l = lcm(&x, &y);
        let z: f64 = 5.0;
        assert_eq!(l.interpolate(z), 2.879078879932948);

        let x: Vec<f64> = vec![1.0, 3.0, 6.0, 10.0, 11.0, 13.0, 17.0, 20.0];
        for (x_i, mu_i) in x.into_iter().zip(mu.into_iter()) {
            assert_eq!(l.interpolate(x_i), mu_i);
        }

        let z: f64 = -1.0;
        assert_eq!(l.interpolate(z), 1.194370974562763);

        let z: f64 = 25.0;
        assert_eq!(l.interpolate(z), -0.0012692901211508456);
    }

    #[test]
    fn gcm_example_2_works() {
        let (x, y, mu, _) = example_2();
        let g = gcm_ltor(x, y);
        assert_eq!(g.mu(), &mu);
    }

    #[test]
    fn lcm_example_2_works() {
        let (x, y, _, mu) = example_2();
        let l = lcm(&x, &y);
        assert_eq!(l.mu(), &mu);
    }
}

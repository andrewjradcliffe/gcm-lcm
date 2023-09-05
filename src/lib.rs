use std::iter::zip;

/// The result of greatest convex minorant construction, which occurs
/// via `gcm` or `gcm_unordered`.
#[derive(Debug, Clone)]
pub struct Gcm {
    x: Vec<f64>,
    f: Vec<f64>,
    dfdx: Vec<f64>,
}
impl Gcm {
    /// Return the value of the greatest convex minorant at `x`. If `x` is outside
    /// the domain of the inputs, then this is extrapolation.
    pub fn interpolate(&self, x: f64) -> f64 {
        match self.x.binary_search_by(|x_j| x_j.total_cmp(&x)) {
            Ok(j) => self.f[j],
            Err(j) => {
                let k = self.x.len();
                if j == 0 {
                    // Below left boundary, extrapolate using derivative at boundary.
                    self.f[0] + self.dfdx[0] * (x - self.x[0])
                } else if j == k {
                    // Above right boundary, extrapolate using derivative at
                    // preceding point; we do not know the derivative at the boundary.
                    self.f[k - 2] + self.dfdx[k - 2] * (x - self.x[k - 2])
                } else {
                    // x[j - 1] < x < x[j]
                    // At or below right boundary, this is just the recurrence
                    // relation iterated to an intermediate point, hence,
                    // it is fully rigorous.
                    self.f[j - 1] + self.dfdx[j - 1] * (x - self.x[j - 1])
                }
            }
        }
    }

    /// Return the value of the derivative of the greatest convex minorant at `x`.
    /// If `x` is outside the domain defined by the inputs, then this
    /// is clamped to the appropriate end of the codomain of the derivative.
    pub fn derivative(&self, x: f64) -> f64 {
        let k = self.x.len();
        match self.x.binary_search_by(|x_j| x_j.total_cmp(&x)) {
            Ok(j) => {
                if j == k - 1 {
                    // At right boundary, we do not know the derivative, hence, clamp.
                    self.dfdx[k - 2]
                } else {
                    self.dfdx[j]
                }
            }
            Err(j) => {
                if j == 0 {
                    self.dfdx[0]
                } else if j >= k - 1 {
                    // At the right boundary, hence, clamp.
                    self.dfdx[k - 2]
                } else {
                    // The derivative between blocks is the derivative of the left block,
                    // not a linear interpolation between the two derivatives.
                    // While it is arguably correct for the implicit function
                    // (i.e. the derivative) to be interpolated, it is not consistent
                    // with the definition of f(i) = f(i-1) + Df(i-1) * (x(i) - x(i-1))
                    // which imposes a constant derivative between i and i-1.
                    self.dfdx[j - 1]
                }
            }
        }
    }

    /// Return the domain.
    pub fn x<'a>(&'a self) -> &'a Vec<f64> {
        &self.x
    }

    /// Return the codomain (computed by the algorithm).
    pub fn f<'a>(&'a self) -> &'a Vec<f64> {
        &self.f
    }

    /// Return the codomain of the derivative.
    pub fn dfdx<'a>(&'a self) -> &'a Vec<f64> {
        &self.dfdx
    }
}

/// Construct the greatest convex minorant of the sequence of points
/// *(xᵢ, yᵢ), i = 0,...,n-1*, assuming that
/// (1) the values satisfy *xᵢ < xᵢ₊₁* for *i = 0,...,n-2*,
/// (2) *-∞ < xᵢ < ∞ ∀i*, and
/// (3) *xᵢ* is not NaN *∀i*.
/// The result of the algorithm is essentially meaningless if these
/// three conditions are not satisfied; however, the implementation
/// will not panic.
/// In addition to the three conditions above, `n` must be at least 2,
/// and `x.len() == y.len()` must hold. Failure to satisfy these two
/// conditions will result in a panic.
///
/// If uniqueness and order of the *x*-values cannot be guaranteed,
/// then `gcm_unordered` should be used. On the other hand,
/// if the guarantees can be upheld, `gcm` should be used as it
/// eliminates a few allocations and guarantees *O*(*n*) time complexity.
///
/// # Examples
/// ```
/// use gcm_lcm::gcm;
/// let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 7.0, 8.0, 9.0];
/// let y: Vec<f64> = vec![1.0, 3.0, 2.0, 5.0, 6.0, 5.0, 7.5];
/// let g = gcm(&x, &y);
/// assert_eq!(g.f(), &vec![1.0, 1.5, 2.0, 2.6, 4.4, 5.0, 7.5]);
/// ```
pub fn gcm(x: &[f64], y: &[f64]) -> Gcm {
    gcm_ltor(x.to_vec(), y.to_vec())
}

// This assumes that z is ordered by the first element of the pairs.
// Note that though we use `total_cmp` elsewhere, the use of equality
// here is correct insofar as we do not have any `NaN`s -- in which
// case, the algorithm is undefined.
fn dedup_by_min(z: Vec<(f64, f64)>) -> (Vec<f64>, Vec<f64>) {
    let n = z.len();
    let mut x: Vec<f64> = Vec::with_capacity(n);
    let mut y: Vec<f64> = Vec::with_capacity(n);
    let mut i: usize = 0;
    while i < n {
        let a = z[i].0;
        let mut b = z[i].1;
        i += 1;
        while i < n && z[i].0 == a {
            b = b.min(z[i].1);
            i += 1;
        }
        x.push(a);
        y.push(b);
    }
    (x, y)
}

/// Construct the greatest convex minorant of the sequence of points,
/// *(xᵢ, yᵢ), i = 0,...,n-1*, assuming that
/// (1) *-∞ < xᵢ < ∞ ∀i*, and
/// (2) *xᵢ* is not NaN *∀i*.
/// In contrast to `gcm`, it is not assumed that `x` is ordered such that *xᵢ < xᵢ₊₁*,
/// nor that the elements of `x` are unique. Points in the ordered sequence
/// for which *xᵢ = xᵢ₊₁* will be de-duplicated by pairing the unique *x*-value
/// with the minimum of the respective *y*-values.
/// In addition to the conditions above, `x.len() == y.len()` must hold
/// and the number of unique points (after applying the de-duplication above)
/// must be at least 2.
///
/// In choosing between `gcm` and `gcm_unordered`, it should be noted that
/// the latter will incur a few more allocations, and, due to the need to sort
/// the points, has *O*(*n* * log*n*) worst-case time complexity.
pub fn gcm_unordered(x: &[f64], y: &[f64]) -> Gcm {
    assert_eq!(x.len(), y.len());
    let mut z: Vec<_> = x.iter().cloned().zip(y.iter().cloned()).collect();
    z.sort_unstable_by(|(a, _), (b, _)| a.total_cmp(b));
    let (x, y) = dedup_by_min(z);
    gcm_ltor(x, y)
}

fn diff(x: &[f64]) -> Vec<f64> {
    x.windows(2).map(|w| w[1] - w[0]).collect()
}

fn gcm_ltor(x: Vec<f64>, y: Vec<f64>) -> Gcm {
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
            let nu_prime = nu[j - 1] + nu[j];
            let xi_prime = xi[j - 1] + xi[j];
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
    let mut dfdx = v;
    let mut f_prev = f[0];
    let mut i: usize = 1;
    for (nu_j, (xi_j, w_j)) in zip(nu, zip(xi, w)) {
        let dfdx_j = nu_j / xi_j;
        for (f_i, (dx_i, dfdx_i)) in f[i..i + w_j].iter_mut().zip(
            dx[i - 1..i - 1 + w_j]
                .into_iter()
                .zip(dfdx[i - 1..i - 1 + w_j].iter_mut()),
        ) {
            *dfdx_i = dfdx_j;
            *f_i = f_prev + dfdx_j * *dx_i;
            f_prev = f_i.clone();
        }
        i += w_j;
    }
    Gcm { x, f, dfdx }
}

/// The result of least concave minorant construction, which occurs
/// via `lcm` or `lcm_unordered`.
#[derive(Debug, Clone)]
pub struct Lcm {
    g: Gcm,
}
impl Lcm {
    /// Return the value of the greatest convex minorant at `x`. If `x` is outside
    /// the domain of the inputs, then this is extrapolation.
    pub fn interpolate(&self, x: f64) -> f64 {
        self.g.interpolate(x)
    }

    /// Return the value of the derivative of the greatest convex minorant at `x`.
    /// If `x` is outside the domain defined by the inputs, then this
    /// is clamped to the appropriate end of the codomain of the derivative.
    pub fn derivative(&self, x: f64) -> f64 {
        self.g.derivative(x)
    }

    /// Return the domain.
    pub fn x<'a>(&'a self) -> &'a Vec<f64> {
        &self.g.x
    }

    /// Return the codomain (computed by the algorithm).
    pub fn f<'a>(&'a self) -> &'a Vec<f64> {
        &self.g.f
    }

    /// Return the codomain of the derivative.
    pub fn dfdx<'a>(&'a self) -> &'a Vec<f64> {
        &self.g.dfdx
    }
}

/// Construct the least concave majorant of the sequence of points
/// *(xᵢ, yᵢ), i = 0,...,n-1*, assuming that
/// (1) the values satisfy *xᵢ < xᵢ₊₁* for *i = 0,...,n-2*,
/// (2) *-∞ < xᵢ < ∞ ∀i*, and
/// (3) *xᵢ* is not NaN *∀i*.
/// The result of the algorithm is essentially meaningless if these
/// three conditions are not satisfied; however, the implementation
/// will not panic.
/// In addition to the three conditions above, `n` must be at least 2,
/// and `x.len() == y.len()` must hold. Failure to satisfy these two
/// conditions will result in a panic.
///
/// If uniqueness and order of the *x*-values cannot be guaranteed,
/// then `lcm_unordered` should be used. On the other hand,
/// if the guarantees can be upheld, `lcm` should be used as it
/// eliminates a few allocations and guarantees *O*(*n*) time complexity.
///
/// # Examples
/// ```
/// use gcm_lcm::lcm;
/// let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 7.0, 8.0, 9.0];
/// let y: Vec<f64> = vec![1.0, 3.0, 2.0, 5.0, 6.0, 5.0, 7.5];
/// let g = lcm(&x, &y);
/// assert_eq!(g.f(), &vec![1.0, 3.0, 4.0, 5.0, 6.5, 7.0, 7.5]);
/// ```
pub fn lcm(x: &[f64], y: &[f64]) -> Lcm {
    let x = x.to_vec();
    let y: Vec<f64> = y.iter().map(|y_i| -*y_i).collect();
    let mut g = gcm_ltor(x, y);
    g.f.iter_mut().for_each(|f_i| *f_i = -*f_i);
    g.dfdx.iter_mut().for_each(|dfdx_i| *dfdx_i = -*dfdx_i);
    Lcm { g }
}

/// Construct the least concave minorant of the sequence of points,
/// *(xᵢ, yᵢ), i = 0,...,n-1*, assuming that
/// (1) *-∞ < xᵢ < ∞ ∀i*, and
/// (2) *xᵢ* is not NaN *∀i*.
/// In contrast to `lcm`, it is not assumed that `x` is ordered such that *xᵢ < xᵢ₊₁*,
/// nor that the elements of `x` are unique. Points in the ordered sequence
/// for which *xᵢ = xᵢ₊₁* will be de-duplicated by pairing the unique *x*-value
/// with the maximum of the respective *y*-values.
/// In addition to the conditions above, `x.len() == y.len()` must hold
/// and the number of unique points (after applying the de-duplication above)
/// must be at least 2.
///
/// In choosing between `lcm` and `lcm_unordered`, it should be noted that
/// the latter will incur a few more allocations, and, due to the need to sort
/// the points, has *O*(*n* * log*n*) worst-case time complexity.
pub fn lcm_unordered(x: &[f64], y: &[f64]) -> Lcm {
    assert_eq!(x.len(), y.len());
    let mut z: Vec<_> = x.iter().cloned().zip(y.iter().map(|y_i| -*y_i)).collect();
    z.sort_unstable_by(|(a, _), (b, _)| a.total_cmp(b));
    let (x, y) = dedup_by_min(z);
    let mut g = gcm_ltor(x, y);
    g.f.iter_mut().for_each(|f_i| *f_i = -*f_i);
    g.dfdx.iter_mut().for_each(|dfdx_i| *dfdx_i = -*dfdx_i);
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

        let f_gcm: Vec<f64> = vec![
            1.755940276352825,
            1.3378194316497374,
            1.5936432121160244,
            1.934741586071074,
            2.0200161795598364,
            2.1905653665373612,
            2.531663740492411,
            2.787487520958698,
        ];
        let f_lcm: Vec<f64> = vec![
            1.755940276352825,
            2.3175095781428867,
            3.159863530827979,
            4.283002134408102,
            4.308393487722174,
            4.359176194350319,
            4.460741607606607,
            2.787487520958698,
        ];
        (x, y, f_gcm, f_lcm)
    }

    fn example_2() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 7.0];
        let y: Vec<f64> = vec![1.0, 3.0, 2.0, 5.0, 6.0];
        let f_gcm: Vec<f64> = vec![1.0, 1.5, 2.0, 3.0, 6.0];
        let f_lcm: Vec<f64> = vec![1.0, 3.0, 4.0, 5.0, 6.0];
        (x, y, f_gcm, f_lcm)
    }

    fn example_3() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let x: Vec<f64> = vec![1.0, 2.0];
        let y: Vec<f64> = vec![1.0, 4.0];
        let f_gcm: Vec<f64> = vec![1.0, 4.0];
        let f_lcm: Vec<f64> = vec![1.0, 4.0];
        (x, y, f_gcm, f_lcm)
    }

    fn example_4() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let x: Vec<f64> = vec![1.0, 2.0, 3.0];
        let y: Vec<f64> = vec![1.0, 4.0, 9.0];
        let f_gcm: Vec<f64> = vec![1.0, 4.0, 9.0];
        let f_lcm: Vec<f64> = vec![1.0, 5.0, 9.0];
        (x, y, f_gcm, f_lcm)
    }

    fn example_5() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 7.0, 8.0];
        let y: Vec<f64> = vec![1.0, 3.0, 2.0, 5.0, 6.0, 5.0];
        let f_gcm: Vec<f64> = vec![1.0, 1.5, 2.0, 13.0 / 5.0, 22.0 / 5.0, 5.0];
        let f_lcm: Vec<f64> = vec![1.0, 3.0, 4.0, 5.0, 6.0, 5.0];
        (x, y, f_gcm, f_lcm)
    }
    fn example_6() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let (mut x, y, f_gcm, f_lcm) = example_5();
        x.iter_mut().for_each(|x_i| *x_i *= 0.5);
        (x, y, f_gcm, f_lcm)
    }
    fn example_7() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let (mut x, y, f_gcm, f_lcm) = example_5();
        x.iter_mut().for_each(|x_i| *x_i *= 0.25);
        (x, y, f_gcm, f_lcm)
    }
    fn example_8() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let (mut x, y, f_gcm, f_lcm) = example_7();
        x.iter_mut().for_each(|x_i| *x_i *= 1.5);
        (x, y, f_gcm, f_lcm)
    }
    fn example_9() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 7.0, 8.0];
        let y: Vec<f64> = vec![1.0, -3.0, -5.0, -7.0, -8.0, -12.0];
        let f_gcm: Vec<f64> = vec![1.0, -3.0, -5.0, -7.0, -10.75, -12.0];
        let f_lcm: Vec<f64> = vec![1.0, -0.5, -2.0, -3.5, -8.0, -12.0];
        (x, y, f_gcm, f_lcm)
    }
    fn example_10() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 7.0, 8.0];
        let y: Vec<f64> = vec![1.0, -3.0, -5.0, -7.0, -8.0, 5.0];
        let f_gcm: Vec<f64> = vec![1.0, -3.0, -5.0, -7.0, -8.0, 5.0];
        let f_lcm: Vec<f64> = vec![
            1.0,
            1.5714285714285714, // 1.5714285714285716
            2.142857142857143,  // 2.1428571428571432
            2.7142857142857144, // 2.714285714285715
            4.428571428571429,
            5.0,
        ];
        (x, y, f_gcm, f_lcm)
    }

    #[test]
    fn gcm_example_1_interpolation_works() {
        let (x, y, f, _) = example_1();
        let g = gcm_ltor(x, y);
        let z: f64 = 5.0;
        assert_eq!(g.interpolate(z), 1.5083686186272622);

        let x: Vec<f64> = vec![1.0, 3.0, 6.0, 10.0, 11.0, 13.0, 17.0, 20.0];
        for (x_i, f_i) in x.into_iter().zip(f.into_iter()) {
            assert_eq!(g.interpolate(x_i), f_i);
        }

        let z: f64 = -1.0;
        assert_eq!(g.interpolate(z), 2.1740611210559124);

        let z: f64 = 25.0;
        assert_eq!(g.interpolate(z), 3.21386048840251);
    }

    #[test]
    fn lcm_example_1_interpolation_works() {
        let (x, y, _, f) = example_1();
        let l = lcm(&x, &y);
        let z: f64 = 5.0;
        assert_eq!(l.interpolate(z), 2.8790788799329485);

        let x: Vec<f64> = vec![1.0, 3.0, 6.0, 10.0, 11.0, 13.0, 17.0, 20.0];
        for (x_i, f_i) in x.into_iter().zip(f.into_iter()) {
            assert_eq!(l.interpolate(x_i), f_i);
        }

        let z: f64 = -1.0;
        assert_eq!(l.interpolate(z), 1.1943709745627633);

        let z: f64 = 25.0;
        assert_eq!(l.interpolate(z), -0.0012692901211508456);
    }

    macro_rules! gcmlcm_example {
        { $test:ident $example:ident } => {
            #[test]
            fn $test() {
                let (x, y, f, _) = $example();
                let g = gcm_ltor(x, y);
                assert_eq!(g.f(), &f);

                let (x, y, _, f) = $example();
                let l = lcm(&x, &y);
                assert_eq!(l.f(), &f);
            }
        }
    }

    gcmlcm_example! { gcmlcm_example_1_works example_1 }
    gcmlcm_example! { gcmlcm_example_2_works example_2 }
    gcmlcm_example! { gcmlcm_example_3_works example_3 }
    gcmlcm_example! { gcmlcm_example_4_works example_4 }
    gcmlcm_example! { gcmlcm_example_5_works example_5 }
    gcmlcm_example! { gcmlcm_example_6_works example_6 }
    gcmlcm_example! { gcmlcm_example_7_works example_7 }
    gcmlcm_example! { gcmlcm_example_8_works example_8 }
    gcmlcm_example! { gcmlcm_example_9_works example_9 }
    gcmlcm_example! { gcmlcm_example_10_works example_10 }

    #[test]
    fn gcm_inf_behavior() {
        let inf = f64::INFINITY;
        let x: Vec<f64> = vec![1.0, 2.0, 3.0];
        let y: Vec<f64> = vec![1.0, 4.0, inf];
        let g = gcm(&x, &y);
        assert_eq!(g.f(), &vec![1.0, 4.0, inf]);
        assert_eq!(g.interpolate(3.0), inf);
        assert_eq!(g.interpolate(4.0), inf);

        let x: Vec<f64> = vec![1.0, 2.0, 3.0];
        let y: Vec<f64> = vec![1.0, 4.0, -inf];
        let g = gcm(&x, &y);
        assert_eq!(g.f(), &vec![1.0, -inf, -inf]);
        assert_eq!(g.interpolate(3.0), -inf);
        assert!(!g.interpolate(2.5).is_nan());

        let x: Vec<f64> = vec![1.0, 2.0, 3.0];
        let y: Vec<f64> = vec![1.0, inf, 9.0];
        let g = gcm(&x, &y);

        let lhs = g.f();
        assert_eq!(lhs[0], 1.0);
        assert!(lhs[1].is_nan());
        assert!(lhs[2].is_nan());

        let x: Vec<f64> = vec![1.0, 2.0, 3.0];
        let y: Vec<f64> = vec![inf; 3];
        let g = gcm(&x, &y);
        assert_eq!(g.f()[0], inf);
        assert!(g.f()[1..].iter().all(|f_i| f_i.is_nan()));
        assert!(g.dfdx().iter().all(|f_i| f_i.is_nan()));

        let x: Vec<f64> = vec![1.0, 2.0, 3.0];
        let y: Vec<f64> = vec![f64::MAX, f64::MAX, inf];
        let g = gcm(&x, &y);
        assert_eq!(g.f(), &y);
        assert_eq!(g.dfdx(), &vec![0.0, inf]);
    }

    #[test]
    fn gcm_nan_behavior() {
        let x: Vec<f64> = vec![1.0, f64::NAN, 3.0];
        let y: Vec<f64> = vec![1.0, 4.0, 9.0];
        let g = gcm(&x, &y);
        assert!(g.f().iter().any(|f_i| f_i.is_nan()));

        assert_eq!(g.interpolate(1.0), 1.0);
        assert!(g.interpolate(2.0).is_nan());
        assert!(g.interpolate(3.0).is_nan());

        let x: Vec<f64> = vec![1.0, 2.0, 3.0];
        let y: Vec<f64> = vec![1.0, f64::NAN, 9.0];
        let g = gcm(&x, &y);
        assert!(g.f().iter().any(|f_i| f_i.is_nan()));
        assert_eq!(g.interpolate(1.0), 1.0);
        assert!(g.interpolate(2.0).is_nan());
        assert!(g.interpolate(3.0).is_nan());

        let x: Vec<f64> = vec![1.0, 2.0, 3.0];
        let y: Vec<f64> = vec![f64::NAN; 3];
        let g = gcm(&x, &y);
        assert!(g.f().iter().all(|f_i| f_i.is_nan()));
        assert!(g.dfdx().iter().all(|f_i| f_i.is_nan()));
    }

    #[test]
    fn derivative_works() {
        let x: Vec<f64> = vec![
            1.8155, 2.4122, 2.4455, 2.9653, 3.1504, 3.8246, 3.8406, 4.2418, 4.2586, 4.9758,
        ];
        let y: Vec<f64> = vec![
            0.0989, 0.1678, 0.1710, 0.1993, 0.1972, 1.3035, 1.2431, 0.8232, 0.7623, 0.0283,
        ];
        let g = gcm(&x, &y);
        let dfdx_2: Vec<f64> = x[0..x.len() - 1].iter().map(|&x| g.derivative(x)).collect();
        assert_eq!(g.dfdx(), &dfdx_2);

        // Is it clamped?
        assert_eq!(g.derivative(5.0), g.derivative(4.9758));
        assert_eq!(g.derivative(1.0), g.derivative(1.8155));
        assert_eq!(g.derivative(4.96), g.derivative(4.9758));

        let l = lcm(&x, &y);
        let dfdx_2: Vec<f64> = x[0..x.len() - 1].iter().map(|&x| l.derivative(x)).collect();
        assert_eq!(l.dfdx(), &dfdx_2);

        assert_eq!(l.derivative(5.0), l.derivative(4.9758));
        assert_eq!(l.derivative(1.0), l.derivative(1.8155));
        assert_eq!(g.derivative(4.96), g.derivative(4.9758));
    }
    #[test]
    fn derivative_is_self_consistent() {
        let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 7.0, 8.0, 9.0];
        let y: Vec<f64> = vec![1.0, 3.0, 2.0, 5.0, 6.0, 5.0, 8.0];
        let g = gcm(&x, &y);
        // Is it consistent?
        // x=3.0 is one of the discontinuities in the derivative. Thus, this tests
        // for properties we expect across adjacent blocks.
        let x_tilde: f64 = 2.9;
        let eps: f64 = 0.5;
        let x_left: f64 = 3.0;
        let dfdx = (g.interpolate(x_tilde + eps) - g.interpolate(x_tilde)) / eps;
        // The derivative should not be equal irrespective of whether we interpolate.
        assert_ne!(dfdx, g.derivative(x_tilde));
        // This should be the case irrespective of whether we interpolate or not
        // since these are in different blocks.
        assert_ne!(g.derivative(x_tilde + eps), g.derivative(x_tilde));
        // This is the self-consistent definition; it is merely a check.
        let fx = g.interpolate(x_left) + g.derivative(x_left) * (x_tilde + eps - x_left);
        assert_eq!(fx, g.interpolate(x_tilde + eps));

        // This remains within a single block, hence, the derivative should be equal.
        // If the derivative were interpolated, this would fail.
        let x_tilde: f64 = 2.7;
        let eps: f64 = 0.2;
        let x_left: f64 = 2.0;
        let dfdx = (g.interpolate(x_tilde + eps) - g.interpolate(x_tilde)) / eps;
        // Within a block, these should be equal. If we interpolated, this would fail.
        assert_eq!(g.derivative(x_tilde + eps), g.derivative(x_tilde));
        assert_eq!(g.derivative(x_tilde), g.derivative(x_left));
        // The forward difference, computed on two points within the same block,
        // should be equal to the derivative.
        assert!((dfdx - g.derivative(x_tilde)).abs() <= 2.0 * f64::EPSILON);
        // Note the difference: the derivative is wrt an interior point.
        let fx = g.interpolate(x_left) + g.derivative(x_tilde) * (x_tilde + eps - x_left);
        assert_eq!(fx, g.interpolate(x_tilde + eps));
    }

    #[test]
    fn dedup_by_min_works() {
        let z: Vec<(f64, f64)> = vec![(1.0, 2.0), (1.0, 1.0), (2.0, 3.0), (2.0, 2.5)];
        assert_eq!(dedup_by_min(z), (vec![1.0, 2.0], vec![1.0, 2.5]));

        let z: Vec<(f64, f64)> = vec![(1.0, 2.0), (1.0, 2.0), (2.0, 3.0), (2.0, 3.0)];
        assert_eq!(dedup_by_min(z), (vec![1.0, 2.0], vec![2.0, 3.0]));

        let z: Vec<(f64, f64)> = vec![(1.0, 2.0), (1.0, 1.0), (2.0, 3.0), (2.0, 2.5), (3.0, -5.0)];
        assert_eq!(dedup_by_min(z), (vec![1.0, 2.0, 3.0], vec![1.0, 2.5, -5.0]));

        let z: Vec<(f64, f64)> = vec![
            (1.0, 5.0),
            (1.0, 1.0),
            (1.0, 2.0),
            (1.0, 1.0),
            (2.0, 3.0),
            (2.0, 2.5),
            (3.0, -5.0),
        ];
        assert_eq!(dedup_by_min(z), (vec![1.0, 2.0, 3.0], vec![1.0, 2.5, -5.0]));

        let z: Vec<(f64, f64)> = vec![
            (1.0, 5.0),
            (1.0, 1.0),
            (1.0, 2.0),
            (1.0, 1.0),
            (2.0, 3.0),
            (2.0, 2.5),
            (3.0, -5.0),
            (3.0, 4.0),
            (3.0, -100.0),
            (3.0, -50.0),
        ];
        assert_eq!(
            dedup_by_min(z),
            (vec![1.0, 2.0, 3.0], vec![1.0, 2.5, -100.0])
        );
    }

    #[test]
    fn gcm_lcm_unordered_works() {
        let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 7.0, 8.0, 9.0];
        let y: Vec<f64> = vec![1.0, 3.0, 2.0, 5.0, 6.0, 5.0, 8.0];
        let f_gcm: Vec<f64> = vec![1.0, 1.5, 2.0, 13.0 / 5.0, 22.0 / 5.0, 5.0, 8.0];
        // The last two values are 37.0 / 5.0, 8.0, but finite precision forces this.
        let f_lcm: Vec<f64> = vec![
            1.0,
            3.0,
            4.0,
            5.0,
            34.0 / 5.0,
            7.3999999999999995,
            7.999999999999999,
        ];

        // First, do we replicate `gcm` and `lcm` as expected?
        let g = gcm_unordered(&x, &y);
        assert_eq!(g.f(), &f_gcm);
        let l = lcm_unordered(&x, &y);
        assert_eq!(l.f(), &f_lcm);

        let x: Vec<f64> = x.into_iter().rev().collect();
        let y: Vec<f64> = y.into_iter().rev().collect();

        // Second: simple reverse ordering
        let g = gcm_unordered(&x, &y);
        assert_eq!(g.f(), &f_gcm);
        let l = lcm_unordered(&x, &y);
        assert_eq!(l.f(), &f_lcm);

        // Now, duplicates
        let x: Vec<f64> = vec![4.0, 8.0, 1.0, 2.0, 3.0, 3.0, 3.0, 4.0, 7.0, 8.0, 9.0];
        let y: Vec<f64> = vec![5.0, 5.0, 1.0, 3.0, 2.5, 3.0, 2.0, 5.0, 6.0, 5.0, 8.0];
        let g = gcm_unordered(&x, &y);
        assert_eq!(g.f(), &f_gcm);
        let l = lcm_unordered(&x, &y);
        assert_eq!(l.f(), &f_lcm);

        let x: Vec<f64> = vec![9.0, 1.0, 2.0, 3.0, 4.0, 7.0, 8.0, 9.0, 3.0, 3.0, 4.0];
        let y: Vec<f64> = vec![10.0, 1.0, 3.0, 2.0, 5.0, 6.0, 5.0, 8.0, 2.5, 3.0, 6.0];
        let g = gcm_unordered(&x, &y);
        assert_eq!(g.f(), &f_gcm);
        let l = lcm_unordered(&x, &y);
        assert_eq!(
            l.f(),
            &vec![
                1.0,
                3.0,
                4.5,
                6.0,
                8.4,
                9.200000000000001,
                10.000000000000002
            ]
        );
    }

    fn is_primal_feasible(x: &[f64]) -> bool {
        x.windows(2).all(|w| w[0] <= w[1])
    }

    fn verify_kkt_conditions(x: Vec<f64>, y: Vec<f64>, slackness_tol: f64, slackness_tol2: f64) {
        let df = diff(&y);
        let dx = diff(&x);
        let dfdx: Vec<f64> = df.iter().zip(dx.iter()).map(|(df, dx)| *df / *dx).collect();

        let g = gcm(&x, &y);

        let dfbardx: Vec<f64> = g.dfdx().clone();
        // Primal feasibility
        assert!(is_primal_feasible(&dfbardx));

        // Dual feasibility
        // This is slightly awkward without access to the `w` from the solver.
        // However, with the derivative from the solver, it can be exact.
        let n = dfbardx.len();
        let mut i: usize = 0;
        while i < n {
            let a = dfbardx[i];
            let mut b = dfdx[i] * dx[i];
            let mut c = dfbardx[i] * dx[i];
            i += 1;
            assert_eq!((0.0_f64).min(b - c), 0.0);
            while i < n && dfbardx[i] == a {
                // N.B. The condition is the `assert!`, but the `assert_eq!` is
                // equivalent and gives a more informative error message.
                // assert!(b - c >= 0.0);
                // assert_eq!((0.0_f64).min(b - c), 0.0);
                b += dfdx[i] * dx[i];
                c += dfbardx[i] * dx[i];
                i += 1;
                assert!((0.0_f64).min(b - c).abs() < 1e-15);
            }
        }

        // Complementary slackness
        let mut lambda: Vec<f64> = vec![0.0; n];
        i = 1;
        // lambda[-1] is zero, but, for convenience of not reindexing, we omit it.
        lambda[0] = (dfdx[0] - dfbardx[0]) * dx[0];
        while i < n {
            lambda[i] = lambda[i - 1] + (dfdx[i] - dfbardx[i]) * dx[i];
            i += 1;
        }
        i = 0;
        // An absolute tolerance of f64::EPSILON works for most test cases,
        // but under the conditions out in the wild, i.e. poorly-scaled data
        // which yields very large forward differences, we must accommodate
        // a less stringent absolute tolerance.
        let eps = slackness_tol;
        while i < n - 1 {
            // This is the condition, but we must accommodate finite precision.
            // assert_eq!(lambda[i] * (dfbardx[i] - dfbardx[i + 1]), 0.0);
            assert!((lambda[i] * (dfbardx[i] - dfbardx[i + 1])).abs() < eps);
            i += 1;
        }

        // Alternate method to compute the Lagrange multipliers
        // and thereby verify both dual feasibility and complementary slackness.
        // It is worthwhile to use both approaches, as this method enables a tighter
        // check of complementary slackness, but requires a looser check of dual
        // feasibility.
        let fbar: Vec<f64> = g.f().clone();
        let lambda: Vec<f64> = y[1..]
            .iter()
            .zip(fbar[1..].iter())
            .map(|(y_i, f_i)| *y_i - *f_i)
            .collect();
        // Dual feasibility
        let mut i: usize = 0;
        while i < n {
            assert!((0.0_f64).min(lambda[i]).abs() < 1e-13);
            i += 1;
        }
        // Complementary slackness
        let eps = slackness_tol2;
        let mut i: usize = 0;
        while i < n - 1 {
            // The condition, but, as above, we must accommodate finite precision.
            // assert_eq!(lambda[i] * (dfbardx[i] - dfbardx[i + 1]), 0.0);
            assert!((lambda[i] * (dfbardx[i] - dfbardx[i + 1])).abs() < eps);
            i += 1;
        }
    }

    #[test]
    fn verify_kkt_1() {
        let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 7.0, 8.0, 9.0];
        let y: Vec<f64> = vec![1.0, 3.0, 2.0, 5.0, 6.0, 5.0, 8.0];
        verify_kkt_conditions(x, y, 3.0 * f64::EPSILON, f64::EPSILON);
    }
    #[test]
    fn verify_kkt_2() {
        let x: Vec<f64> = vec![1.0, 3.0, 6.0, 10.0, 11.0, 13.0, 17.0, 20.0];
        let y: Vec<f64> = vec![-2.45, 10.86, 3.91, 8.14, 9.29, 17.19, 13.3, 24.1];
        verify_kkt_conditions(x, y, f64::EPSILON, f64::EPSILON);
    }
    #[test]
    fn verify_kkt_3() {
        let x: Vec<f64> = vec![0.42, 4.49, 4.71, 10.02, 12.41, 14.88, 16.98, 19.16];
        let y: Vec<f64> = vec![-2.45, 10.86, 3.91, 8.14, 9.29, 17.19, 13.3, 24.1];
        verify_kkt_conditions(x, y, 50.0 * f64::EPSILON, 40.0 * f64::EPSILON);
    }
    #[test]
    fn verify_kkt_4() {
        let x: Vec<f64> = vec![
            1.8155, 2.4122, 2.4455, 2.9653, 3.1504, 3.8246, 3.8406, 4.2418, 4.2586, 4.9758,
        ];
        let y: Vec<f64> = vec![
            0.0989, 0.1678, 0.1710, 0.1993, 0.1972, 1.3035, 1.2431, 0.8232, 0.7623, 0.0283,
        ];
        verify_kkt_conditions(x, y, f64::EPSILON, f64::EPSILON);
    }
    #[test]
    fn verify_kkt_5() {
        let x: Vec<f64> = vec![0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17];
        let y: Vec<f64> = vec![1.0, 10.0, -1.0, -10.0, 10.0, -10.0, 0.0, 0.0];
        verify_kkt_conditions(x, y, 10000.0 * f64::EPSILON, f64::EPSILON);
    }
    #[test]
    fn verify_kkt_6() {
        let x: Vec<f64> = vec![
            -1.0, 10000.0, 50000.0, 100000.0, 100001.0, 120001.0, 121001.0, 200001.0,
        ];
        let y: Vec<f64> = vec![
            0.234, 51.355, 118.267, 198.133, 223.487, 335.363, 1000.357, 1005.121,
        ];
        verify_kkt_conditions(x, y, f64::EPSILON, f64::EPSILON);
    }
    #[test]
    fn verify_kkt_7() {
        let x: Vec<f64> = vec![
            0.234, 51.355, 118.267, 198.133, 223.487, 335.363, 1000.357, 1005.121,
        ];
        let y: Vec<f64> = vec![
            -1.0, 10000.0, 50000.0, 100000.0, 100001.0, 120001.0, 121001.0, 200001.0,
        ];
        // The second slackness tolerance applies only to the last term.
        verify_kkt_conditions(x, y, f64::EPSILON, 1e-6);
    }

    macro_rules! verify_kkt {
        { $test:ident $example:ident } => {
            #[test]
            fn $test() {
                let (x, y, _, _) = $example();
                verify_kkt_conditions(x, y, f64::EPSILON, f64::EPSILON);
            }
        }
    }
    verify_kkt! { kkt_example_1 example_1 }
    verify_kkt! { kkt_example_2 example_2 }
    verify_kkt! { kkt_example_3 example_3 }
    verify_kkt! { kkt_example_4 example_4 }
    verify_kkt! { kkt_example_5 example_5 }
    verify_kkt! { kkt_example_6 example_6 }
    verify_kkt! { kkt_example_7 example_7 }
    verify_kkt! { kkt_example_8 example_8 }
    verify_kkt! { kkt_example_9 example_9 }
    verify_kkt! { kkt_example_10 example_10 }
}

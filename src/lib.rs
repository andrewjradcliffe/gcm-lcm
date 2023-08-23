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

fn diff(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut dx: Vec<f64> = Vec::with_capacity(n - 1);

    for i in 0..n - 1 {
        // We know this is in bounds given that the length is n, and
        // (n - 2 + 1) = n - 1 is the last offset accessed.
        let delta = unsafe { *x.get_unchecked(i + 1) - *x.get_unchecked(i) };
        dx.push(delta);
    }
    dx
}

pub fn gcm(x: Vec<f64>, y: Vec<f64>) -> Gcm {
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
    let mut w: Vec<usize> = Vec::with_capacity(k);
    w.resize(k, 1);
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
pub fn lcm(x: Vec<f64>, y: Vec<f64>) -> Lcm {
    let mut y = y;
    y.iter_mut().for_each(|y_i| *y_i = -*y_i);
    let mut g = gcm(x, y);
    g.mu.iter_mut().for_each(|mu_i| *mu_i = -*mu_i);
    Lcm { g }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn example_1() -> (Vec<f64>, Vec<f64>) {
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
        (x, y)
    }

    #[test]
    fn gcm_example_1_works() {
        let mu: Vec<f64> = vec![
            1.755940276352825,
            1.3378194316497374,
            1.5936432121160244,
            1.934741586071074,
            2.0200161795598364,
            2.1905653665373612,
            2.531663740492411,
            2.787487520958698,
        ];
        let (x, y) = example_1();
        let g = gcm(x, y);
        assert_eq!(g.mu(), &mu);
    }
    #[test]
    fn gcm_example_1_interpolation_works() {
        let (x, y) = example_1();
        let g = gcm(x, y);
        let z: f64 = 5.0;
        assert_eq!(g.interpolate(z), 1.508368618627262);

        let mu: Vec<f64> = vec![
            1.755940276352825,
            1.3378194316497374,
            1.5936432121160244,
            1.934741586071074,
            2.0200161795598364,
            2.1905653665373612,
            2.531663740492411,
            2.787487520958698,
        ];
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
        let mu: Vec<f64> = vec![
            1.755940276352825,
            2.3175095781428867,
            3.159863530827979,
            4.283002134408102,
            4.308393487722174,
            4.359176194350319,
            4.460741607606607,
            2.787487520958698,
        ];
        let (x, y) = example_1();
        let l = lcm(x, y);
        assert_eq!(l.mu(), &mu);
    }

    #[test]
    fn lcm_example_1_interpolation_works() {
        let (x, y) = example_1();
        let l = lcm(x, y);
        let z: f64 = 5.0;
        assert_eq!(l.interpolate(z), 2.879078879932948);

        let mu: Vec<f64> = vec![
            1.755940276352825,
            2.3175095781428867,
            3.159863530827979,
            4.283002134408102,
            4.308393487722174,
            4.359176194350319,
            4.460741607606607,
            2.787487520958698,
        ];
        let x: Vec<f64> = vec![1.0, 3.0, 6.0, 10.0, 11.0, 13.0, 17.0, 20.0];
        for (x_i, mu_i) in x.into_iter().zip(mu.into_iter()) {
            assert_eq!(l.interpolate(x_i), mu_i);
        }

        let z: f64 = -1.0;
        assert_eq!(l.interpolate(z), 1.194370974562763);

        let z: f64 = 25.0;
        assert_eq!(l.interpolate(z), -0.0012692901211508456);
    }
}

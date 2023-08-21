#[derive(Debug)]
pub struct Gcm {
    pub x: Vec<f64>,
    pub mu: Vec<f64>,
}
fn diff(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut dx: Vec<f64> = Vec::with_capacity(n - 1);
    // let mut iter = x.iter();
    // if let Some(x_i_m1) = iter.next() {
    //     let mut x_i = x_i_m1.clone();
    //     iter.map(|x_i_m1| {
    //         let delta = x_i - *x_i_m1;
    //         x_i = x_i_m1.clone();
    //         delta
    //     })
    //     .collect()
    // }

    for i in 0..n - 1 {
        // We know this is in bounds given that the length is n, and
        // (n - 2 + 1) = n - 1 is the last offset accessed.
        let delta = unsafe { *x.get_unchecked(i + 1) - *x.get_unchecked(i) };
        dx.push(delta);
    }
    dx
}

pub fn gcm(x: Vec<f64>, y: Vec<f64>) -> Gcm {
    // These two necessary conditions could be handled more delicately.
    let n = y.len();
    assert_eq!(x.len(), n);
    assert!(n > 1);

    let mut nu = diff(&y);
    let mut dx = diff(&x);
    let k = dx.len();
    let mut w: Vec<usize> = Vec::with_capacity(k);
    w.resize(n, 1);
    loop {
        let k = nu.len();
        let mut j = k - 1;
        // while j > 0 {
        //     if nu[j - 1] / dx[j - 1] > nu[j] / dx[j] {
        //         break;
        //     }
        //     j -= 1;
        // }
        while j > 0 && nu[j - 1] / dx[j - 1] <= nu[j] / dx[j] {
            j -= 1;
        }
        if j == 0 {
            let mut nu_out: Vec<f64> = Vec::with_capacity(k);

            // Safe due to satisfaction of necessary conditions
            // nu_out.push(unsafe { y.get_unchecked(0).clone() });
            nu_out.push(y[0]);
            let mut pos: usize = 1;
            let k = nu.len();
            for i in 0..k {
                let mu = nu[i] / dx[i];
                for _ in 0..w[i] {
                    nu_out.push(nu_out[pos - 1] + mu * (x[pos] - x[pos - 1]));
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
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
        let g = gcm(x, y);
        assert_eq!(g.mu, mu);
    }
}

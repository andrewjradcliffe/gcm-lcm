# gcm-lcm

[![crate](https://img.shields.io/crates/v/gcm-lcm.svg)](https://crates.io/crates/gcm-lcm)
[![documentation](https://docs.rs/gcm-lcm/badge.svg)](https://docs.rs/gcm-lcm)

## Usage
Add this to your `Cargo.toml`:

```toml
[dependencies]
gcm-lcm = "0.1"
```

## Description

Construct the greatest convex minorant (GCM) or least concave majorant
(LCM). This may be of use if your computations involve stochastic
processes. Alternatively, perhaps you wish to construct a convex (or
concave) function approximation for use in an optimization problem,
simulation (e.g. inverse transform sampling using the LCM of an
empirical cumulative distribution function), or something more
creative.

The full description of the algorithm is provided in a
[pre-print](https://andrewjradcliffe.github.io/gcm-algorithm.pdf).

## License

Licensed under either of

  * [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0)
  * [MIT license](http://opensource.org/licenses/MIT)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.

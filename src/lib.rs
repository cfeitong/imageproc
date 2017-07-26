#![feature(inclusive_range_syntax)]
#![feature(iterator_for_each)]
#![warn(unused_qualifications)]
#![allow(dead_code)]
// #![allow(unused_imports)]

extern crate libc;
extern crate nalgebra;
extern crate num;

#[macro_use]
pub mod image;
pub mod traits;
pub mod imageio;
pub mod convert;
pub mod transform;
pub mod geo;
pub mod math;
pub mod op;
pub mod eye;
pub mod pixel;

pub use imageio::{ImageIO, FreeImageIO};

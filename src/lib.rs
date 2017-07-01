#![feature(inclusive_range_syntax)]
#![warn(unused_qualifications)]
#![allow(dead_code)]

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

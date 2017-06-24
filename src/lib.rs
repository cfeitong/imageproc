#![feature(inclusive_range_syntax)]
#![warn(unused_qualifications)]
#![allow(dead_code)]

extern crate libc;
extern crate nalgebra;
extern crate num;

mod traits;
pub mod image;
pub mod imageio;
pub mod convert;
pub mod transform;
pub mod geo;
pub mod math;
pub mod op;

pub use imageio::{ImageIO, FreeImageIO};

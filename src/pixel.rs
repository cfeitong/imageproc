use traits::Primitive;
use num::{Saturating, NumCast};
use std::ops::{Index, IndexMut, Add, Sub, Mul};
use std::fmt::Debug;

#[derive(PartialEq, Eq, Clone, Debug, Copy, Hash)]
#[repr(C, packed)]
pub struct Color<T: Primitive> {
    b: T,
    g: T,
    r: T,
    a: T,
}

impl<T: Primitive> Color<T> {
    fn new(b: T, g: T, r: T, a: T) -> Color<T> {
        Color {
            b: b,
            g: g,
            r: r,
            a: a,
        }
    }

    fn from_gray(gray: T) -> Color<T> {
        Color {
            b: gray,
            g: gray,
            r: gray,
            a: T::max_value(),
        }
    }

    fn is_gray(&self) -> bool {
        (self.b == self.g) && (self.r == self.g)
    }
}

pub const MAX_CHANNEL_COUNT: usize = 4;

/// A pixel object is usually not used standalone but as a view into an image buffer.
pub trait Pixel: Debug + Copy + Clone + Index<usize> {
    /// The underlying subpixel type.
    type Subpixel: Primitive;

    fn zero() -> Self;

    fn from_raw(data: &[Self::Subpixel]) -> Self;

    /// Returns the number of channels of this pixel type.
    fn channels() -> usize;

    /// Returns the bits per pixel
    fn bits_per_pixel() -> usize;

    /// Returns the components as a slice.
    fn raw(&self) -> &[Self::Subpixel];

    /// Returns the components as a mutable slice
    fn raw_mut(&mut self) -> &mut [Self::Subpixel];

    fn blend(self, other: Self, alpha: f32) -> Self;
    fn blend4(self, b: Self, c: Self, d: Self, u: f32, v: f32) -> Self;
}

// originally from https://github.com/PistonDevelopers/image
macro_rules! define_colors {
    {$(
        $ident:ident,
        $channels: expr,
        $interpretation: expr,
        #[$doc:meta];
    )*} => {

$( // START Structure definitions

#[$doc]
#[derive(PartialEq, Eq, Clone, Debug, Copy, Hash)]
#[repr(C, packed)]
#[allow(missing_docs)]
pub struct $ident<T: Primitive> { pub data: [T; $channels] }

#[allow(non_snake_case, dead_code)]
#[inline]
pub fn $ident<T: Primitive>(data: [T; $channels]) -> $ident<T> {
        $ident {
                    data: data
        }
}

impl<T: Primitive> Pixel for $ident<T> {
    type Subpixel = T;

    #[inline]
    fn zero() -> $ident<T> {
        $ident {
            data: [T::zero(); $channels]
        }
    }

    #[inline]
    fn from_raw(data: &[T]) -> Self {
        let mut t = $ident::zero();
        for i in 0..$channels { t.data[i] = data[i]; }
        t
    }

    #[inline]
    fn channels() -> usize {
        $channels
    }

    #[inline]
    fn bits_per_pixel() -> usize {
        8 * $channels
    }

    #[inline]
    fn raw(&self) -> &[T] {
        &self.data
    }

    #[inline]
    fn raw_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    #[inline]
    fn blend(self, other: Self, alpha: f32) -> Self {
        // OPTIMIZE
        let mut t = self.data;
        for i in 0..$channels {
            t[i] = NumCast::from(t[i].to_f32().unwrap() * alpha
                + other.data[i].to_f32().unwrap() * (1f32 - alpha)).unwrap();
        }
        $ident(t)
    }

    #[inline]
    fn blend4(self, b: Self, c: Self, d: Self, u: f32, v: f32) -> Self {
        let a0 = u * v;
        let a1 = (1.0 - u) * v;
        let a2 = u * (1.0 - v);
        let a3 = (1.0 - u) * (1.0 - v);

        let mut a = self.data;
        for i in 0..$channels {
            a[i] = NumCast::from(
                a[i].to_f32().unwrap() * a0
                + b[i].to_f32().unwrap() * a1
                + c[i].to_f32().unwrap() * a2
                + d[i].to_f32().unwrap() * a3
                ).unwrap();
        }
        $ident(a)
    }
}

impl<T: Primitive> Index<usize> for $ident<T> {
    type Output = T;

    #[inline]
    fn index(&self, _index: usize) -> &T {
        &self.data[_index]
    }
}

impl<T: Primitive> IndexMut<usize> for $ident<T> {
    #[inline]
    fn index_mut(&mut self, _index: usize) -> &mut T {
        &mut self.data[_index]
    }
}

impl<T: Primitive> Add for $ident<T> {
    type Output = $ident<T>;

    #[inline]
    fn add(self, other: $ident<T>) -> $ident<T> {
        let mut m = [T::zero(); $channels];
        for i in 0..$channels { m[i] = self.data[i] + other.data[i]; }
        $ident(m)
    }
}

impl<T: Primitive> Sub for $ident<T> {
    type Output = $ident<T>;

    #[inline]
    fn sub(self, other: $ident<T>) -> $ident<T> {
        let mut m = [T::zero(); $channels];
        for i in 0..$channels { m[i] = self.data[i] - other.data[i]; }
        $ident(m)
    }
}

impl<T: Primitive> Mul for $ident<T> {
    type Output = $ident<T>;

    #[inline]
    fn mul(self, other: $ident<T>) -> $ident<T> {
        let mut m = [T::zero(); $channels];
        for i in 0..$channels { m[i] = self.data[i] * other.data[i]; }
        $ident(m)
    }
}

// Pixel + scalar
impl<T: Primitive, U: Primitive> Add<U> for $ident<T> {
    type Output = $ident<T>;

    #[inline]
    fn add(self, other: U) -> $ident<T> {
        let mut m = [T::zero(); $channels];
        for i in 0..$channels {
            let a = self.data[i].to_f32().unwrap();
            let b = other.to_f32().unwrap();
            m[i] = if T::max_value().to_f32().unwrap() < a+b {
                T::max_value()
            } else {
                T::from(a + b).unwrap()
            };

            // m[i] = U::from(self.data[i]).unwrap() + other;
        }
        $ident(m)
    }
}

// Pixel - scalar
impl<T: Primitive, U: Primitive> Sub<U> for $ident<T> {
    type Output = $ident<T>;

    #[inline]
    fn sub(self, other: U) -> $ident<T> {
        let mut m = [T::zero(); $channels];
        for i in 0..$channels {
            let a = self.data[i].to_f32().unwrap();
            let b = other.to_f32().unwrap();

            m[i] = if T::min_value().to_f32().unwrap() > a-b {
                T::min_value()
            } else {
                T::from(a - b).unwrap()
            };
            // m[i] = U::from(self.data[i]).unwrap() - other;
        }
        $ident(m)
    }
}

// Pixel * scalar
impl<T: Primitive, U: Primitive> Mul<U> for $ident<T> {
    type Output = $ident<T>;

    #[inline]
    fn mul(self, other: U) -> $ident<T> {
        let mut m = [T::zero(); $channels];
        for i in 0..$channels {
            let a = self.data[i].to_f32().unwrap();
            let b = other.to_f32().unwrap();

            m[i] = if T::max_value().to_f32().unwrap() < a * b {
                T::max_value()
            } else {
                T::from(a * b).unwrap()
            }
            // m[i] = U::from(self.data[i]).unwrap() * other;
        }
        $ident(m)
    }
}

)* // END
    }
}

define_colors! {
    BGR, 3, "BGR", #[doc = "RGB colors"];
    Gray, 1, "Y", #[doc = "GrayScale colors"];
    BGRA, 4, "BGRA", #[doc = "BGR colors + alpha channel"];
    RGBA, 4, "RGBA", #[doc = "RGB colors + alpha channel"];
}

pub trait AlphaPixel: Pixel {
    fn alpha_index() -> usize;
}

macro_rules! define_alpha(
    ($t:ident, $idx:expr) => (
impl<T: Primitive> AlphaPixel for $t<T> {
    fn alpha_index() -> usize {
        $idx
    }
}
    );
);

define_alpha!(BGRA, 3);
define_alpha!(RGBA, 3);

pub trait RGBPixel: Pixel {
    fn red_index() -> usize;
    fn green_index() -> usize;
    fn blue_index() -> usize;
}

macro_rules! define_rgb(
    ($t:ident, $r:expr, $g:expr, $b:expr) => (
impl<T: Primitive> RGBPixel for $t<T> {
    fn blue_index() -> usize {
        $b
    }
    fn green_index() -> usize {
        $g
    }
    fn red_index() -> usize {
        $r
    }
}
    );
);

define_rgb!(BGR, 2, 1, 0);
define_rgb!(BGRA, 2, 1, 0);
define_rgb!(RGBA, 0, 1, 2);

macro_rules! define_saturating(
    ($t:ident) => (
impl<T: Primitive + Saturating> Saturating for $t<T> {
    #[inline]
    fn saturating_add(self, other: $t<T>) -> $t<T> {
        let mut m = self.data;
        for i in 0..$t::<T>::channels() as usize {
            m[i] = m[i].saturating_add(other.data[i]); }
        $t(m)
    }
    #[inline]
    fn saturating_sub(self, other: $t<T>) -> $t<T> {
        let mut m = self.data;
        for i in 0..$t::<T>::channels() as usize {
            m[i] = m[i].saturating_sub(other.data[i]); }
        $t(m)
    }
}
);
);

define_saturating!(Gray);
define_saturating!(BGR);
define_saturating!(BGRA);

// Gray pixel literal
pub fn gray<T: Primitive>(x: T) -> Gray<T> {
    let t = [x,];
    Gray::from_raw(&t)
}

// BGR pixel literal
pub fn bgr<T: Primitive>(b: T, g: T, r: T) -> BGR<T> {
    let t = [b, g, r];
    BGR::from_raw(&t)
}

// BGRA pixel literal
pub fn bgra<T: Primitive>(b: T, g: T, r: T, a: T) -> BGRA<T> {
    let t = [b, g, r, a];
    BGRA::from_raw(&t)
}


#[cfg(test)]
mod test {
    use std::mem;
    use super::*;

    #[test]
    fn test_pixel_size() {
        assert_eq!(mem::size_of::<BGRA<u8>>(), 4);
        assert_eq!(mem::size_of::<BGR<u8>>(), 3);
        assert_eq!(mem::size_of::<Gray<u8>>(), 1);
    }

}

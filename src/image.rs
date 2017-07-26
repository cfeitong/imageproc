use std::slice;
use num::traits::Bounded;
use std::ops::{Index, IndexMut};
use pixel::{Pixel, AlphaPixel, BGR, BGRA, Gray, Binary};

#[derive(Debug)]
pub enum ImageError {
    InvalidImage,
    OutOfRegion,
    OutOfMemoryError,
    UnknownImageFormat,
    UnknownError,
}

pub trait GenericImage {
    type Pixel: Pixel;
}

#[derive(PartialEq, Eq, Debug)]
pub struct Image<T: Pixel> {
    w: usize,
    h: usize,
    stride: usize, //stride in sizeof(T)
    data: Vec<T>,
}

impl<T: Pixel> GenericImage for Image<T> {
    type Pixel = T;
}

impl<'a, T: Pixel + 'a> Image<T> {
    pub fn new(width: usize, height: usize) -> Image<T> {
        // fast allocation without initialization
        let len = (width as usize) * (height as usize);
        let mut data: Vec<T> = Vec::with_capacity(len);
        unsafe {
            data.set_len(len);
        }
        Image {
            w: width,
            h: height,
            stride: width,
            data: data.clone(),
        }
    }

    #[inline]
    pub fn width(&self) -> usize {
        self.w
    }

    #[inline]
    pub fn height(&self) -> usize {
        self.h
    }

    #[inline]
    pub fn size(&self) -> (usize, usize) {
        (self.w, self.h)
    }

    #[inline]
    pub fn stride(&self) -> usize {
        self.stride
    }

    #[inline]
    pub fn pitch(&self) -> usize {
        self.stride * (self.bits_per_pixel() / 8) as usize
    }

    #[inline]
    pub fn pixels(&self) -> &[T] {
        self.data.as_slice()
    }

    #[inline]
    pub fn pixels_mut(&mut self) -> &mut [T] {
        self.data.as_mut_slice()
    }

    pub fn raw(&self) -> &[T::Subpixel] {
        let raw_len = self.bytes_per_row() * self.h as usize;
        unsafe { slice::from_raw_parts(self.data.as_ptr() as *const T::Subpixel, raw_len) }
    }

    pub fn raw_mut(&mut self) -> &mut [T::Subpixel] {
        let raw_len = self.bytes_per_row() * self.h as usize;
        unsafe {
            slice::from_raw_parts_mut(self.pixels_mut().as_mut_ptr() as *mut T::Subpixel, raw_len)
        }
    }

    #[inline]
    pub fn channels(&self) -> usize {
        T::channels()
    }

    #[inline]
    pub fn bits_per_pixel(&self) -> usize {
        T::bits_per_pixel()
    }

    #[inline]
    pub fn bytes_per_row(&self) -> usize {
        (self.stride as usize) * (T::bits_per_pixel() / 8) as usize
    }

    #[inline]
    pub fn row(&self, r: usize) -> &[T] {
        let start = r * self.stride;
        &self.data[start as usize..(start + self.stride) as usize]
    }

    #[inline]
    pub fn row_mut(&mut self, r: usize) -> &mut [T] {
        let start = r * self.stride;
        &mut self.data[start as usize..(start + self.stride) as usize]
    }

    pub fn fill(&mut self, v: &T) {
        for p in self.data.as_mut_slice().iter_mut() {
            *p = *v;
        }
    }

    pub fn fill_channel(&mut self, ch_idx: usize, v: T::Subpixel) {
        assert!(ch_idx < T::channels() as usize);
        for p in self.data.as_mut_slice().iter_mut() {
            p.raw_mut()[ch_idx] = v;
        }
    }

    pub fn zero(&mut self) {
        self.fill(&T::zero())
    }

    pub fn iter(&self) -> ImageIterator<T> {
        ImageIterator {
            image: &self,
            row: self.row(0),
            y: 0,
            x: 0,
        }
    }

    pub fn iter_mut(&mut self) -> ImageMutIterator<T> {
        ImageMutIterator {
            image: self,
            y: 0,
            x: 0,
        }
    }

    pub fn from_raw(width: usize, height: usize, data: &[T]) -> Self {
        assert_eq!(width * height, data.len());
        let mut img = Self::new(width, height);
        img.data = data.to_vec();
        img
    }
}

pub trait AlphaImage {
    fn set_alpha(&mut self);
}

impl<T: AlphaPixel> AlphaImage for Image<T> {
    fn set_alpha(&mut self) {
        self.fill_channel(T::alpha_index(), T::Subpixel::max_value());
    }
}

impl<T: Pixel> Drop for Image<T> {
    fn drop(&mut self) {
        self.data.clear();
    }
}

impl<T: Pixel> Clone for Image<T> {
    fn clone(&self) -> Image<T> {
        Image {
            w: self.w,
            h: self.h,
            stride: self.stride,
            data: self.data.clone(),
        }
    }
}

impl<T: Pixel> Index<(usize, usize)> for Image<T> {
    type Output = T;

    #[inline]
    fn index(&self, _index: (usize, usize)) -> &T {
        let (x, y) = _index;
        let off = self.stride * y + x;
        &self.data[off as usize]
    }
}

impl<T: Pixel> IndexMut<(usize, usize)> for Image<T> {
    #[inline]
    fn index_mut(&mut self, _index: (usize, usize)) -> &mut T {
        let (x, y) = _index;
        let off = self.stride * y + x;
        &mut self.data[off as usize]
    }
}


pub type ImageGray = Image<Gray<u8>>;
pub type ImageBGR = Image<BGR<u8>>;
pub type ImageBGRA = Image<BGRA<u8>>;
pub type ImageBinary = Image<Binary>;

pub type ImageGrayf = Image<Gray<f32>>;
pub type ImageBGRf = Image<BGR<f32>>;
pub type ImageBGRAf = Image<BGRA<f32>>;

pub struct ImageIterator<'a, P>
where
    P: Pixel + 'a,
    <P as Index<usize>>::Output: 'a,
    P::Subpixel: 'a,
{
    image: &'a Image<P>,
    row: &'a [P],
    y: usize,
    x: usize,
}

impl<'a, P> Iterator for ImageIterator<'a, P>
where
    P: Pixel + 'a,
    <P as Index<usize>>::Output: 'a,
    P::Subpixel: 'a,
{
    type Item = (usize, usize, &'a P);
    #[inline]
    fn next(&mut self) -> Option<(usize, usize, &'a P)> {
        if self.x >= self.image.width() {
            self.y += 1;
            if self.y >= self.image.height() {
                return None;
            } else {
                self.row = self.image.row(self.y);
            }
            self.x = 0;
        }
        let (x, y) = (self.x, self.y);
        self.x += 1;
        Some((x, y, &self.row[x as usize]))
    }
}

pub struct ImageMutIterator<'a, P>
where
    P: Pixel + 'a,
    <P as Index<usize>>::Output: 'a,
    P::Subpixel: 'a,
{
    image: &'a mut Image<P>,
    y: usize,
    x: usize,
}

impl<'a, P> Iterator for ImageMutIterator<'a, P>
where
    P: Pixel + 'a,
    <P as Index<usize>>::Output: 'a,
    P::Subpixel: 'a,
{
    type Item = (usize, usize, &'a mut P);
    #[inline]
    fn next(&mut self) -> Option<(usize, usize, &'a mut P)> {
        if self.x >= self.image.width() {
            self.y += 1;
            self.x = 0;
        }
        if self.y >= self.image.height() {
            return None;
        }
        let (x, y) = (self.x, self.y);
        self.x += 1;

        // TODO: implement this without `unsafe'
        // seems impossible or very inconvenient at least
        unsafe {
            let t: *mut P = &mut self.image.row_mut(y)[x as usize];
            Some((x, y, &mut *t))
        }
    }
}

impl ImageBinary {
    pub fn invert(&mut self) {
        self.iter_mut().for_each(|(_, _, p)| p.invert());
    }
}

use std::ops::{Not, BitAnd, BitOr, BitXor, BitAndAssign, BitOrAssign, BitXorAssign};

impl Not for ImageBinary {
    type Output = Self;
    fn not(mut self) -> Self::Output {
        for r in 0..self.width() {
            for c in 0..self.height() {
                self[(r, c)] = !self[(r, c)];
            }
        }
        self
    }
}

impl BitAnd for ImageBinary {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self::Output {
        let mut ret = self.clone();
        for x in 0..ret.width() {
            for y in 0..ret.height() {
                ret[(x, y)] = ret[(x, y)] & rhs[(x, y)];
            }
        }
        ret
    }
}

impl BitOr for ImageBinary {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output {
        let mut ret = self.clone();
        for x in 0..ret.width() {
            for y in 0..ret.height() {
                ret[(x, y)] = ret[(x, y)] | rhs[(x, y)];
            }
        }
        ret
    }
}

impl BitXor for ImageBinary {
    type Output = Self;
    fn bitxor(self, rhs: Self) -> Self::Output {
        let mut ret = self.clone();
        for x in 0..ret.width() {
            for y in 0..ret.height() {
                ret[(x, y)] = ret[(x, y)] ^ rhs[(x, y)];
            }
        }
        ret
    }
}

impl BitAndAssign for ImageBinary {
    fn bitand_assign(&mut self, rhs: Self) {
        *self = self.clone() & rhs;
    }
}

impl BitOrAssign for ImageBinary {
    fn bitor_assign(&mut self, rhs: Self) {
        *self = self.clone() | rhs;
    }
}

impl BitXorAssign for ImageBinary {
    fn bitxor_assign(&mut self, rhs: Self) {
        *self = self.clone() ^ rhs;
    }
}


// useful for testing
#[macro_export]
macro_rules! gray_image {
    () => {{
            use image::{ImageGray};
            use pixel::{Gray, gray};

            ImageGray::new(0, 0);
    }};
    ($( $( $x: expr ),*);*) => {{
            use image::{ImageGray};
            use pixel::{gray};

            let nested_array = [ $( [ $($x),* ] ),* ];
            let flat_array: Vec<_> = nested_array.into_iter()
                .flat_map(|row| row.into_iter())
                .cloned()
                .map(|x| gray(x))
                .collect();
            let w = nested_array[0].len();
            let h = nested_array.len();
            ImageGray::from_raw(w, h, &flat_array)

    }}
}


#[cfg(test)]
mod test {
    use super::*;
    use pixel::*;

    #[test]
    fn test_alloc() {
        let img = ImageBGRA::new(100, 200);
        assert_eq!(img.channels(), 4);
        assert_eq!(img.bits_per_pixel(), 4 * 8);
        assert_eq!(img.pixels().len(), 100 * 200);
        assert_eq!(img.raw().len(), 100 * 200 * 4);
        assert_eq!(img.pitch(), 100 * 4);
    }

    #[test]
    fn test_iter() {
        let mut img = ImageBGRA::new(10, 5);
        for (_, _, p) in img.iter_mut() {
            *p = BGRA::<u8>([128, 128, 0, 0]);
        }

        for (_, _, p) in img.iter() {
            assert_eq!(*p, BGRA::<u8>([128, 128, 0, 0]));
        }
    }

    #[test]
    fn test_traits() {
        let mut img = ImageBGRA::new(10, 5);
        for (_, _, p) in img.iter_mut() {
            *p = *p + BGRA::<u8>([128, 128, 0, 0]);
        }
    }

    #[test]
    fn test_binary_not() {
        let mut img =
            gray_image![
            1, 1, 1;
            0, 0, 0;
            1, 0, 1
        ];
        img = !img;
        assert_eq!(
            img,
            gray_image![
            0, 0, 0;
            1, 1, 1;
            0, 1, 0
        ]
        );
    }

    #[test]
    fn test_binary_and() {
        let img1 =
            gray_image![
            1, 1, 1;
            0, 1, 0;
            1, 0, 1
        ];
        let img2 =
            gray_image![
            0, 0, 0;
            1, 1, 1;
            0, 1, 0
        ];
        assert_eq!(
            img1 & img2,
            gray_image![
                0, 0, 0;
                0, 1, 0;
                0, 0, 0
            ]
        );
    }

    #[test]
    fn test_binary_and_assign() {
        let mut img =
            gray_image![
            0, 1, 0;
            1, 1, 1;
            0, 0, 0
        ];
        img &=
            gray_image![
            0, 0, 0;
            0, 1, 0;
            0, 1, 0
        ];
        assert_eq!(
            img,
            gray_image![
            0, 0, 0;
            0, 1, 0;
            0, 0, 0
        ]
        );
    }

    #[test]
    fn test_binary_or() {
        let img1 =
            gray_image![
            1, 1, 1;
            0, 1, 0;
            1, 0, 0
        ];
        let img2 =
            gray_image![
            0, 0, 0;
            1, 1, 1;
            0, 1, 0
        ];
        assert_eq!(
            img1 | img2,
            gray_image![
                1, 1, 1;
                1, 1, 1;
                1, 1, 0
            ]
        );
    }

    #[test]
    fn test_binary_or_assign() {
        let mut img =
            gray_image![
            0, 1, 0;
            1, 1, 1;
            0, 0, 0
        ];
        img |=
            gray_image![
            0, 0, 0;
            0, 1, 0;
            0, 1, 0
        ];
        assert_eq!(
            img,
            gray_image![
            0, 1, 0;
            1, 1, 1;
            0, 1, 0
        ]
        );
    }

    #[test]
    fn test_binary_xor() {
        let img1 =
            gray_image![
            1, 1, 1;
            0, 1, 0;
            1, 0, 0
        ];
        let img2 =
            gray_image![
            0, 0, 1;
            1, 1, 1;
            0, 1, 0
        ];
        assert_eq!(
            img1 ^ img2,
            gray_image![
                1, 1, 0;
                1, 0, 1;
                1, 1, 0
            ]
        );
    }

    #[test]
    fn test_binary_xor_assign() {
        let mut img =
            gray_image![
            0, 1, 0;
            1, 1, 1;
            0, 0, 0
        ];
        img ^=
            gray_image![
            0, 0, 0;
            0, 1, 0;
            0, 1, 0
        ];
        assert_eq!(
            img,
            gray_image![
            0, 1, 0;
            1, 0, 1;
            0, 1, 0
        ]
        );
    }

    #[test]
    fn test_binary_invert() {
        let mut img =
            gray_image![
            0, 1, 0;
            1, 1, 1;
            0, 1, 0
        ];
        img.invert();
        assert_eq!(
            img,
            gray_image![
            1, 0, 1;
            0, 0, 0;
            1, 0, 1
        ]
        );
    }
}

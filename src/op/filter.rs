use image::Image;
use pixel::Pixel;
use std::ops::{Index, IndexMut, Mul};
use num::cast;
use num::{Saturating, ToPrimitive, Bounded};
use eye::Eye;
use math::utils::clip_from_f32;

pub trait Filter {
    fn filter<P>(&self, img: &Image<P>) -> Image<P>
    where
        P: Pixel + Saturating;
}

pub trait Kernel
    : Index<(usize, usize), Output = f32> + IndexMut<(usize, usize), Output = f32> + Filter
    {
    fn size(&self) -> (usize, usize);
}


#[derive(Debug, PartialEq)]
pub struct GeneralKernel {
    data: Vec<Vec<f32>>,
    width: usize,
    height: usize,
}

#[derive(Debug, PartialEq)]
pub struct GaussianKernel {
    data: Vec<Vec<f32>>,
    width: usize,
    height: usize,
}

macro_rules! define_kernel {
    ($Kern: ident) => (
impl Index<(usize, usize)> for $Kern {
    type Output = f32;
    fn index(&self, index: (usize, usize)) -> &f32 {
        let (x, y) = index;
        &self.data[y][x]
    }
}

impl IndexMut<(usize, usize)> for $Kern {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut f32 {
        let (x, y) = index;
        &mut self.data[y][x]
    }
}

impl Filter for $Kern {
    fn filter<P>(&self, img: &Image<P>) -> Image<P>
        where P: Pixel + Mul<f32, Output=P> + Saturating {
        let mut ret: Image<P> = Image::new(img.width(), img.height());
        for y in 0..img.height() {
            for x in 0..img.width() {
                ret[(x, y)] = kern_calc_one_pos(self, x as usize, y as usize, img);
            }
        }
        ret
    }
}

impl Kernel for $Kern {
    fn size(&self) -> (usize, usize) {
        (self.width, self.height)
    }

}
    )
}

define_kernel!(GeneralKernel);
define_kernel!(GaussianKernel);

impl GeneralKernel {
    pub fn new(width: usize, height: usize, _data: &[f32]) -> Self {
        assert_eq!(width * height, _data.len());
        let mut data = Vec::new();
        for i in 0..height {
            let mut tmp = Vec::new();
            for j in 0..width {
                tmp.push(_data[i * width + j]);
            }
            data.push(tmp);
        }
        GeneralKernel {
            data,
            width,
            height,
        }
    }
}

impl GaussianKernel {
    pub fn new(size: usize, sigma: f32) -> Self {
        assert_eq!(size & 1, 1);
        let mut data = Vec::new();
        for x in 0..size {
            let mut row = Vec::new();
            for y in 0..size {
                let dx = (x as f32 - (size / 2) as f32).abs();
                let dy = (y as f32 - (size / 2) as f32).abs();
                let mut cur: f32 = -((dx * dx) / (2f32 * sigma * sigma) +
                                         (dy * dy) / (2f32 * sigma * sigma));
                cur = cur.exp();
                row.push(cur);
            }
            data.push(row);
        }

        let mut result = GaussianKernel {
            data: data,
            width: size,
            height: size,
        };

        {
            normalize(&mut result);
        }
        result
    }
}

fn normalize<T: Kernel>(kern: &mut T) {
    let mut sum = 0f32;
    let (w, h) = kern.size();
    for x in 0..w {
        for y in 0..h {
            sum += kern[(x, y)];
        }
    }
    for x in 0..w {
        for y in 0..h {
            kern[(x, y)] /= sum;
        }
    }
}

fn kern_calc_one_pos<K, P>(kern: &K, x: usize, y: usize, img: &Image<P>) -> P
where
    P: Pixel + Saturating,
    K: Kernel,
{
    let (width, height) = kern.size();
    let half_x = width / 2;
    let half_y = height / 2;
    let sx = x as isize - half_x as isize;
    let sy = y as isize - half_y as isize;

    let mut ret: Vec<f32> = Vec::new();
    for _ in 0..img.channels() {
        ret.push(0f32);
    }
    for i in 0..width {
        for j in 0..height {
            let ix = i as isize + sx;
            let iy = j as isize + sy;
            let eye = Eye::new(ix, iy).extend();
            let a = eye.look(&img);
            let b = kern[(i, j)];
            let v: Vec<f32> = a.raw()
                .iter()
                .map(|ref k| k.to_f32().unwrap() * b)
                .collect();
            ret.iter_mut()
                .enumerate()
                .map(|(i, it)| *it += v[i])
                .collect::<Vec<_>>();
        }
    }
    let ret: Vec<_> = ret.iter()
        .map(|v| {
            clip_from_f32(*v, P::Subpixel::min_value(), P::Subpixel::max_value())
        })
        .collect();
    P::from_raw(&ret)
}

#[derive(Debug)]
pub struct MedianFilter {
    pub width: usize,
    pub height: usize,
}

impl Filter for MedianFilter {
    fn filter<P>(&self, img: &Image<P>) -> Image<P>
    where
        P: Pixel + Mul<f32, Output = P> + Saturating,
    {
        let mut ret: Image<P> = Image::new(img.width(), img.height());
        for x in 0..img.width() {
            for y in 0..img.height() {
                ret[(x, y)] = median_filter_calc_one(self, x as usize, y as usize, img);
            }
        }
        ret
    }
}

fn median_filter_calc_one<P: Pixel>(
    filter: &MedianFilter,
    x: usize,
    y: usize,
    img: &Image<P>,
) -> P {
    let mut ret = P::zero();
    for c in 0..img.channels() {
        let mut counter = Vec::new();
        let sx = x as isize - (filter.width / 2) as isize;
        let sy = y as isize - (filter.height / 2) as isize;
        for x in sx..(sx + filter.width as isize) {
            for y in sy..(sy + filter.height as isize) {
                if 0 <= x && x < img.width() as isize && 0 <= y && y < img.height() as isize {
                    let pixel = img[(x as usize, y as usize)];
                    counter.push(pixel.raw()[c]);
                }
            }
        }
        counter.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let len = counter.len();
        ret.raw_mut()[c] = counter[len / 2];
    }
    ret
}

impl MedianFilter {
    pub fn new(width: usize, height: usize) -> Self {
        assert_eq!(width & 1, 1);
        assert_eq!(height & 1, 1);
        MedianFilter { width, height }
    }
}

#[derive(Debug)]
pub struct BoxFilter {
    pub width: usize,
    pub height: usize,
}

impl Filter for BoxFilter {
    fn filter<P>(&self, img: &Image<P>) -> Image<P>
    where
        P: Pixel + Mul<f32, Output = P> + Saturating,
    {
        let mut ret = Image::new(img.width(), img.height());
        for x in 0..img.width() {
            for y in 0..img.height() {
                ret[(x, y)] = box_filter_calc_one(&self, x, y, &img);
            }
        }
        ret
    }
}

fn box_filter_calc_one<P: Pixel>(filter: &BoxFilter, x: usize, y: usize, img: &Image<P>) -> P {
    let mut ret = P::zero();
    let sx = x as isize - filter.width as isize / 2;
    let sy = y as isize - filter.height as isize / 2;
    let size = filter.width as f32 * filter.height as f32;
    for c in 0..img.channels() {
        let mut sum = 0f32;
        for i in sx..(sx + filter.width as isize) {
            for j in sy..(sy + filter.height as isize) {
                let eye = Eye::new(i, j);
                sum += eye.look(img).raw()[c].to_f32().unwrap();
            }
        }
        ret.raw_mut()[c] = cast(sum / size).unwrap();
    }
    ret
}

impl BoxFilter {
    pub fn new(width: usize, height: usize) -> Self {
        assert_eq!(width & 1, 1);
        assert_eq!(height & 1, 1);
        BoxFilter { width, height }
    }
}


#[cfg(test)]
mod test {
    use super::*;
    use imageio::{ImageIO, FreeImageIO};
    use image::ImageBGR;
    use std::path::Path;

    #[test]
    fn test_convolution() {
        let tmp: Vec<f32> = vec![0f32, 0f32, 1f32, 0f32, 0f32, 1f32, 0f32, 0f32, 1f32];
        let kern = GeneralKernel::new(3, 3, &tmp);
        let path = Path::new("./tests/cat.jpg");
        let img: ImageBGR = FreeImageIO::from_path(&path).unwrap();
        let result = kern.filter(&img);
        let path = Path::new("/tmp/test-conv-out2.jpg");
        FreeImageIO::save(&path, &result).unwrap();
    }

    #[test]
    fn test_gaussian_kernel() {
        let kern = GaussianKernel::new(3, 0.8);
        let path = Path::new("./tests/coins_speckle_0.1.tif");
        let img: ImageBGR = FreeImageIO::from_path(&path).unwrap();
        let result = kern.filter(&img);
        let path = Path::new("/tmp/test-gaussian-kern-out1.jpg");
        FreeImageIO::save(&path, &result).unwrap();
    }

    #[test]
    fn test_median_filter() {
        let filter = MedianFilter::new(5, 5);
        let path = Path::new("./tests/coins_salt_pepper_0.1.tif");
        let img: ImageBGR = FreeImageIO::from_path(&path).unwrap();
        let result = filter.filter(&img);
        let path = Path::new("/tmp/test-median-filter-out1.jpg");
        FreeImageIO::save(&path, &result).unwrap();
    }

    #[test]
    fn test_box_filter() {
        let filter = BoxFilter::new(3, 3);
        let path = Path::new("./tests/coins_salt_pepper_0.1.tif");
        let img: ImageBGR = FreeImageIO::from_path(&path).unwrap();
        let result = filter.filter(&img);
        let path = Path::new("/tmp/test-box-filter-out1.jpg");
        FreeImageIO::save(&path, &result).unwrap();
    }
}

use op::filter::{Filter, GeneralKernel};
use imageio::{ImageIO, FreeImageIO};
use image::{Image, ImageBGRA};
use pixel::Pixel;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sobel {}


impl Sobel {
    fn new() -> Self {
        Sobel {}
    }
}

use num::Saturating;
use std::ops::{Mul, Add};
impl Filter for Sobel {
    fn filter<P>(&self, img: &Image<P>) -> Image<P>
    where
        P: Pixel + Saturating,
    {
        let _kernx = [-1f32, 0f32, 1f32, -2f32, 0f32, 2f32, -1f32, 0f32, 1f32];
        let _kerny = [-1f32, -2f32, -1f32, 0f32, 0f32, 0f32, 1f32, 2f32, 1f32];
        let kernx = GeneralKernel::new(3, 3, &_kernx);
        let kerny = GeneralKernel::new(3, 3, &_kerny);
        let imgx = kernx.filter(img);
        let imgy = kerny.filter(img);

        let mut ret: Image<P> = Image::new(img.width(), img.height());
        for i in 0..ret.width() {
            for j in 0..ret.height() {
                ret[(i, j)] = imgx[(i, j)].saturating_add(imgy[(i, j)]);
            }
        }
        ret
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use image::{ImageGray, ImageBGRA};
    use imageio::{FreeImageIO, ImageIO};
    use std::path::Path;
    use std::fs::File;
    use std::io::Write;

    #[test]
    fn test_sobel_filter() {
        let kern = Sobel::new();
        let path = Path::new("./tests/cat.jpg");
        let img: ImageBGRA = FreeImageIO::from_path(&path).unwrap();
        // let img = gray_image!(
        //     1, 1, 1, 1, 1;
        //     1, 1, 1, 1, 1;
        //     1, 1, 1, 1, 1;
        //     1, 1, 1, 1, 1;
        //     1, 1, 1, 1, 1
        // );
        let result = kern.filter(&img);
        let path = Path::new("/tmp/test-sobel-filter.jpg");
        FreeImageIO::save(&path, &result).unwrap();
        // let mut file = File::create("/tmp/sobel-img.txt").unwrap();
        // write!(file, "{:?}", result);
    }

}

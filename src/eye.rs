use pixel::Pixel;
use std::cmp::{max, min};
use geo::Recti;
use image::Image;
use std::default::Default;

pub trait Mask {
    fn is_valid(&self, x: isize, y: isize) -> bool;
}

impl Mask for Recti {
    fn is_valid(&self, x: isize, y: isize) -> bool {
        self.x <= x && x < self.x + self.width && self.y <= y && y < self.y + self.height
    }
}

#[derive(Debug, Clone, Copy)]
pub enum AlterType<P: Pixel> {
    Constant(P),
    Mirror,
    Extend,
}

// TODO: add mask support
#[derive(Debug, Clone, Copy)]
pub struct Eye<P: Pixel> {
    x: isize,
    y: isize,
    alter_type: AlterType<P>,
}

impl<P: Pixel> Eye<P> {
    pub fn new(x: isize, y: isize) -> Self {
        Self::default().x(x).y(y)
    }

    pub fn x(mut self, x: isize) -> Self {
        self.x = x;
        self
    }

    pub fn y(mut self, y: isize) -> Self {
        self.y = y;
        self
    }

    pub fn constant(mut self, pixel: P) -> Self {
        self.alter_type = AlterType::Constant(pixel);
        self
    }

    pub fn mirror(mut self) -> Self {
        self.alter_type = AlterType::Mirror;
        self
    }

    pub fn extend(mut self) -> Self {
        self.alter_type = AlterType::Extend;
        self
    }

    pub fn look(&self, img: &Image<P>) -> P {
        if 0 <= self.x && self.x < img.width() as isize && 0 <= self.y &&
            self.y < img.height() as isize
        {
            return img[(self.x as usize, self.y as usize)];
        }
        match self.alter_type {
            AlterType::Constant(pixel) => pixel,
            AlterType::Extend => {
                let mut x = self.x;
                let mut y = self.y;
                x = min(x, img.width() as isize - 1);
                x = max(x, 0);
                y = min(y, img.height() as isize - 1);
                y = max(y, 0);
                img[(x as usize, y as usize)].clone()
            }
            AlterType::Mirror => {
                let mut x = self.x;
                let mut y = self.y;
                if x < 0 {
                    x = -x;
                };
                if y < 0 {
                    y = -y;
                };
                if x > img.width() as isize {
                    x = img.width() as isize - (x % img.width() as isize);
                };
                if y > img.height() as isize {
                    y = img.height() as isize - (y % img.height() as isize);
                };
                img[(x as usize, y as usize)].clone()
            }
        }
    }
}

impl<P: Pixel> Default for Eye<P> {
    fn default() -> Self {
        Eye {
            x: 0,
            y: 0,
            alter_type: AlterType::Extend,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use pixel::{Gray, gray};


    #[test]
    fn test_eye_look() {
        let img =
            gray_image![
            1u8, 2u8, 3u8;
            4u8, 5u8, 6u8;
            7u8, 8u8, 9u8
        ];
        let eye: Eye<Gray<u8>> = Eye::new(0, 0);
        assert_eq!(eye.look(&img), gray(1u8));
        assert_eq!(eye.x(400).constant(gray(15u8)).look(&img), gray(15u8));
        assert_eq!(eye.constant(gray(15u8)).look(&img), gray(1u8));
        assert_eq!(eye.x(-1).y(1).mirror().look(&img), gray(5u8));
        assert_eq!(eye.extend().x(1).y(-100).look(&img), gray(2u8));
        assert_eq!(eye.extend().x(-100).y(-100).look(&img), gray(1u8));
        assert_eq!(eye.extend().x(100).y(100).look(&img), gray(9u8));
    }
}

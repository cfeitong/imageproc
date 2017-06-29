use pixel::Pixel;
use geo::{Recti};
use std::ops::{Index, IndexMut};

trait Mask {
    fn is_valid(&self, x: isize, y: isize) -> bool;
}

trait View<P: Pixel>: Index<(usize, usize)> + IndexMut<(usize, usize)> {
    fn default_pixel() -> P;
}

impl Mask for Recti {
    fn is_valid(&self, x: isize, y: isize) -> bool {
        self.x <= x && x < self.x + self.width &&
        self.y <= y && y < self.y + self.height
    }
}

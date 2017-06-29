mod types;
pub mod point;
pub mod rect;

pub use self::point::Point;
pub use self::rect::Rect;

pub type Pointi = Point<isize>;
pub type Pointf = Point<f32>;
pub type Recti = Rect<isize>;
pub type Rectf = Rect<f32>;

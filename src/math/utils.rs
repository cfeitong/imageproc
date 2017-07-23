use traits::Primitive;

#[inline]
pub fn clipped_round(x: f32, min: i32, max: i32) -> i32 {
    let t = x.round() as i32;
    if t < min {
        return min;
    }
    if t > max {
        return max;
    }
    t
}

#[inline]
pub fn clip<T: Primitive>(x: T, min: T, max: T) -> T {
    if x < min {
        return min;
    }
    if x > max {
        return max;
    }
    x
}

pub fn clip_from_f32<T: Primitive>(x: f32, min: T, max: T) -> T {
    let min = min.to_f32().unwrap();
    let max = max.to_f32().unwrap();
    let ret = if x < min {
                min
            } else if x > max {
                max
            } else {
                x
            };
    T::from(ret).unwrap()
}
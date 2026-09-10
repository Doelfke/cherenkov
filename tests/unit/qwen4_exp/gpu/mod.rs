//! GPU checks grouped by the subsystem under test.

use super::*;
use half::bf16;

mod attention;
mod qsa;
mod sampling;
mod state;

struct Lcg(u64);

impl Lcg {
    fn f(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn upload<T: Copy>(ctx: &MetalContext, data: &[T]) -> Buf {
    let b = ctx.new_buffer(std::mem::size_of_val(data).max(4)).unwrap();
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr(), b.contents().cast::<T>().as_ptr(), data.len())
    };
    b
}

fn download<T: Copy>(b: &Buf, n: usize) -> Vec<T> {
    unsafe { std::slice::from_raw_parts(b.contents().cast::<T>().as_ptr(), n) }.to_vec()
}

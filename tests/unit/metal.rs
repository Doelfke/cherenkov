use super::*;

#[test]
fn allocation_budget_covers_owned_and_no_copy_buffers() {
    let ctx = MetalContext::new().unwrap();
    let bytes = memmap2::MmapMut::map_anon(PAGE_SIZE).unwrap();
    ctx.allocation_limit
        .set(Some(ctx.device.currentAllocatedSize()));
    assert!(ctx.new_buffer(PAGE_SIZE).is_err());
    assert!(unsafe { ctx.wrap_mmap(&bytes) }.is_err());
    assert!(unsafe { ctx.wrap_region(bytes.as_ptr() as *mut u8, PAGE_SIZE) }.is_err());
    ctx.allocation_limit.set(None);
    assert!(ctx.new_buffer(PAGE_SIZE).is_ok());
}

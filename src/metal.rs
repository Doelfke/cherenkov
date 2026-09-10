use anyhow::{Context, Result};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary,
    MTLResourceOptions, MTLSize,
};
use std::ffi::c_void;
use std::ptr::NonNull;
use std::time::Instant;

pub const PAGE_SIZE: usize = 16384;

pub struct MetalContext {
    pub allocation_limit: std::cell::Cell<Option<usize>>,
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
}

impl MetalContext {
    pub fn new() -> Result<Self> {
        let device = MTLCreateSystemDefaultDevice().context("no Metal device")?;
        let queue = device.newCommandQueue().context("newCommandQueue failed")?;
        Ok(MetalContext {
            device,
            queue,
            allocation_limit: std::cell::Cell::new(None),
        })
    }

    pub fn compile_library(
        &self,
        source: &str,
    ) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>> {
        let source = NSString::from_str(source);
        self.device
            .newLibraryWithSource_options_error(&source, None)
            .map_err(|e| anyhow::anyhow!("MSL compile failed: {e}"))
    }

    pub fn pipeline(
        &self,
        library: &ProtocolObject<dyn MTLLibrary>,
        function: &str,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
        let name = NSString::from_str(function);
        let function_obj = library
            .newFunctionWithName(&name)
            .with_context(|| format!("kernel function {function:?} not found"))?;
        let pso = self
            .device
            .newComputePipelineStateWithFunction_error(&function_obj)
            .map_err(|e| anyhow::anyhow!("pipeline creation failed for {function}: {e}"))?;
        // Register-cliff canary: the compiler lowers a pipeline's
        // maxTotalThreadsPerThreadgroup below the 1024 default exactly when
        // it hits its register budget (Apple family 9). Warn so cliffs are
        // caught at load, not in tokens/s. Keep the warning for unexpectedly narrow pipelines.
        let max_threads = pso.maxTotalThreadsPerThreadgroup();
        if max_threads < 256 {
            eprintln!(
                "WARNING: pipeline {function} register-limited: \
                 max_threads/tg = {max_threads} (< 256)"
            );
        }
        Ok(pso)
    }

    /// Expose an existing mapping as a Metal buffer without copying it.
    ///
    /// # Safety
    /// The page-rounded region must remain mapped until this buffer and
    /// all GPU work using it are finished. The caller must synchronize
    /// CPU writes with GPU access; the buffer does not own the mapping.
    pub unsafe fn wrap_mmap(
        &self,
        bytes: &[u8],
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>> {
        let ptr = bytes.as_ptr() as *mut c_void;
        anyhow::ensure!(
            (ptr as usize).is_multiple_of(PAGE_SIZE),
            "mmap base is not page-aligned"
        );
        let len = bytes.len().div_ceil(PAGE_SIZE) * PAGE_SIZE;
        self.check_allocation(len)?;
        let ptr = NonNull::new(ptr).context("null mmap pointer")?;
        unsafe {
            self.device
                .newBufferWithBytesNoCopy_length_options_deallocator(
                    ptr,
                    len,
                    MTLResourceOptions::empty(),
                    None,
                )
        }
        .context("newBufferWithBytesNoCopy failed (is the region page-aligned?)")
    }

    /// Wrap a page-aligned portion of a mapping owned by the expert pool.
    ///
    /// # Safety
    /// `ptr..ptr+len` must remain mapped for the buffer's lifetime and all
    /// submitted GPU work. CPU/GPU access must obey the pool's event
    /// handshake, including outstanding reads before a slot is reused.
    pub unsafe fn wrap_region(
        &self,
        ptr: *mut u8,
        len: usize,
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>> {
        anyhow::ensure!(
            (ptr as usize).is_multiple_of(PAGE_SIZE),
            "region not page-aligned"
        );
        anyhow::ensure!(
            len.is_multiple_of(PAGE_SIZE),
            "region length not page-aligned"
        );
        let ptr = NonNull::new(ptr.cast::<c_void>()).context("null region")?;
        self.check_allocation(len)?;
        unsafe {
            self.device
                .newBufferWithBytesNoCopy_length_options_deallocator(
                    ptr,
                    len,
                    MTLResourceOptions::empty(),
                    None,
                )
        }
        .context("newBufferWithBytesNoCopy failed for session region")
    }

    fn check_allocation(&self, len: usize) -> Result<()> {
        if let Some(limit) = self.allocation_limit.get() {
            anyhow::ensure!(
                self.device
                    .currentAllocatedSize()
                    .checked_add(len)
                    .is_some_and(|n| n <= limit),
                "allocation would exceed the server memory budget; reduce pool or prefill chunk size"
            );
        }
        Ok(())
    }

    pub fn new_buffer(&self, len: usize) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>> {
        self.check_allocation(len)?;
        self.device
            .newBufferWithLength_options(len, MTLResourceOptions::empty())
            .context("newBufferWithLength failed")
    }

    pub fn throttle_probe(&self) -> Result<f64> {
        let lib = self.compile_library(crate::kernels::CLOCK_PROBE_MSL)?;
        let pso = self.pipeline(&lib, "clock_probe")?;
        let out = self.new_buffer(65536 * 4)?;
        let iters: u32 = 60_000;
        let mut last = 0.0f64;
        for _ in 0..2 {
            let cb = self.queue.commandBuffer().context("commandBuffer")?;
            let enc = cb.computeCommandEncoder().context("encoder")?;
            enc.setComputePipelineState(&pso);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&out), 0, 0);
                enc.setBytes_length_atIndex(
                    NonNull::from(&iters).cast::<c_void>(),
                    size_of::<u32>(),
                    1,
                );
            }
            let grid = MTLSize {
                width: 65536,
                height: 1,
                depth: 1,
            };
            let tg = MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            };
            enc.dispatchThreads_threadsPerThreadgroup(grid, tg);
            enc.endEncoding();
            let start = Instant::now();
            cb.commit();
            cb.waitUntilCompleted();
            last = start.elapsed().as_secs_f64() * 1e3;
        }
        Ok(last)
    }
}

#[cfg(test)]
#[path = "../tests/unit/metal.rs"]
mod tests;

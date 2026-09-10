use super::*;

/// Exercise both reduction stages, including ties across the grid stride,
/// an output in the final partial block, and all-negative-infinity input.
#[test]
fn greedy_argmax_preserves_ties_and_partial_blocks() {
    let ctx = MetalContext::new().unwrap();
    let library = ctx.compile_library(FORWARD_MSL).unwrap();
    let partial = ctx.pipeline(&library, "argmax_partial").unwrap();
    let final_reduce = ctx.pipeline(&library, "argmax_final").unwrap();
    let grid_stride = ARGMAX_TGS * 256;
    let len = grid_stride + 35;

    for scenario in 0..3 {
        let mut values = vec![f32::NEG_INFINITY; len];
        let expected = match scenario {
            0 => {
                values.fill(-10.0);

                for index in [5, 36, 4097, grid_stride + 5] {
                    values[index] = 3.5;
                }

                5
            }
            1 => {
                values[len - 1] = -1.0;

                len - 1
            }
            _ => 0,
        };
        let logits = upload(&ctx, &values);
        // Metal ArgPair is a float followed by a uint, eight bytes total.
        let partials = upload(&ctx, &vec![0u8; ARGMAX_TGS * 8]);
        let ids = upload(&ctx, &[u32::MAX; 3]);
        let cb = ctx.queue.commandBuffer().unwrap();
        let enc = cb.computeCommandEncoder().unwrap();

        enc.setComputePipelineState(&partial);

        unsafe {
            enc.setBuffer_offset_atIndex(Some(&logits), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&partials), 0, 1);
        }

        set_bytes(&enc, 2, &(len as u32));
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: ARGMAX_TGS,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        enc.memoryBarrierWithScope(objc2_metal::MTLBarrierScope::Buffers);
        enc.setComputePipelineState(&final_reduce);

        unsafe {
            enc.setBuffer_offset_atIndex(Some(&partials), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&ids), 0, 1);
        }

        set_bytes(&enc, 2, &(ARGMAX_TGS as u32));
        set_bytes(&enc, 3, &0u32);
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 1024,
                height: 1,
                depth: 1,
            },
        );
        enc.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
        assert_eq!(
            download::<u32>(&ids, 3),
            [u32::MAX, expected as u32, u32::MAX]
        );
    }
}

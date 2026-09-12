use super::*;
use crate::qwen4_exp::{Qwen4ExpConfig, RopeParams};
use objc2_metal::MTLDevice;

fn config() -> Qwen4ExpConfig {
    Qwen4ExpConfig {
        hidden_size: 256,
        num_hidden_layers: 2,
        layer_types: vec!["linear_attention".into(), "full_attention".into()],
        num_attention_heads: 8,
        num_key_value_heads: 2,
        head_dim: 32,
        linear_num_key_heads: 2,
        linear_num_value_heads: 4,
        linear_key_head_dim: 32,
        linear_value_head_dim: 32,
        linear_conv_kernel_dim: 4,
        rms_norm_eps: 1e-6,
        vocab_size: 1024,
        partial_rotary_factor: 0.25,
        rope_parameters: RopeParams {
            rope_theta: 10000.0,
        },
        hc_count: 4,
        hc_lowrank: 32,
        num_experts: 16,
        num_experts_per_tok: 2,
        moe_intermediate_size: 128,
        shared_expert_intermediate_size: 128,
        norm_topk_prob: true,
        ple_layer_ids: vec![],
        ple_conv_kernel_size: 4,
        ple_embed_dim: 32,
        ngram_size: 3,
        heads_per_ngram: 8,
        indexer_n_heads: 2,
        indexer_kv_heads: 1,
        indexer_head_dim: 32,
        indexer_budget: 128,
        indexer_compress_ratio: 4,
        output_gate_type: "sigmoid".into(),
        eos_token_id: 0,
        mtp_num_hidden_layers: 1,
    }
}

#[test]
fn capacity_respects_context_padding_and_exhausted_memory() -> Result<()> {
    let c = config();

    for context in [2048, 120000, 262144] {
        assert!(rows_fit(&c, context, 0, false).is_err());

        let minimum = scratch_bytes(&c, context, 1, false)?;

        assert!(rows_fit(&c, context, minimum - 1, false).is_err());

        for requested in [1, 32, 128, 512, 1024, 4096] {
            let bytes = scratch_bytes(&c, context, requested, false)?;
            let fits = rows_fit(&c, context, bytes, false)?;

            assert!(fits >= requested.min(context));
            assert!(scratch_bytes(&c, context, fits, false)? <= bytes);

            if fits < MAX_PREFILL_ROWS.min(context) {
                assert!(scratch_bytes(&c, context, fits + 1, false)? > bytes);
            }
        }
    }

    Ok(())
}

#[test]
fn capacity_accounts_for_context_logits_and_small_budgets() -> Result<()> {
    let c = config();
    let bytes = scratch_bytes(&c, 2048, 128, false)?;

    assert!(rows_fit(&c, 2048, bytes, false)? < 512);
    assert!(scratch_bytes(&c, 120000, 128, false)? > bytes);
    assert!(scratch_bytes(&c, 2048, 128, true)? > bytes);
    assert!(rows_fit(&c, 2048, bytes, true)? < 128);

    Ok(())
}

#[test]
fn scratch_budget_covers_real_metal_allocations() -> Result<()> {
    let c = config();
    let ctx = MetalContext::new()?;

    for all_logits in [false, true] {
        let budget = scratch_bytes(&c, 2048, 129, all_logits)?;
        let before = ctx.device.currentAllocatedSize();

        ctx.allocation_limit.set(Some(before + budget));

        let buffers = scratch(&c, 2048, 129, all_logits, |n| ctx.new_buffer(n))?;

        assert_eq!(buffers.rows, 192);
        assert!(ctx.device.currentAllocatedSize() - before <= budget);
        assert_eq!(buffers.logits_all.is_some(), all_logits);
        drop(buffers);
    }

    Ok(())
}

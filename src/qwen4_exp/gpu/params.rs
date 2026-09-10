//! Host/shader ABI: preserve field order and match the Metal declarations.

#[repr(C)]
pub(super) struct IndexParams {
    pub(super) ihd: u32,
    pub(super) inh: u32,
    pub(super) qk_dim: u32,
    pub(super) ratio: u32,
    pub(super) rot: u32,
    pub(super) theta: f32,
    pub(super) eps: f32,
    pub(super) base_pos: u32,
    pub(super) nb: u32,
    pub(super) b0: u32,
    pub(super) b1: u32,
    pub(super) k: u32,
    pub(super) max_blocks: u32,
    pub(super) vis_stride: u32,
    pub(super) mask_words: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct GroupParams {
    pub(super) n: u32,
    pub(super) groups: u32,
    pub(super) eps: f32,
    pub(super) shift: f32,
}

#[repr(C)]
pub(super) struct NormPrepParams {
    pub(super) n: u32,
    pub(super) groups: u32,
    pub(super) eps: f32,
    pub(super) inject: u32,
}

#[repr(C)]
pub(super) struct MoeBParams {
    pub(super) inter: u32,
    pub(super) hidden: u32,
    pub(super) n_u: u32,
    pub(super) gate_w: u32,
    pub(super) up_w: u32,
    pub(super) down_w: u32,
    pub(super) gate_s: u32,
    pub(super) gate_b: u32,
    pub(super) up_s: u32,
    pub(super) up_b: u32,
    pub(super) down_s: u32,
    pub(super) down_b: u32,
    pub(super) layer: u32,
    pub(super) shared: u32,
    pub(super) nb: u32,
    pub(super) sh_gate_w: u32,
    pub(super) sh_gate_s: u32,
    pub(super) sh_gate_b: u32,
    pub(super) sh_up_w: u32,
    pub(super) sh_up_s: u32,
    pub(super) sh_up_b: u32,
    pub(super) sh_down_w: u32,
    pub(super) sh_down_s: u32,
    pub(super) sh_down_b: u32,
    pub(super) sh_gate_vec: u32,
    pub(super) part: u32,
    pub(super) low_gate_w: u32,
    pub(super) low_up_w: u32,
    pub(super) low_down_w: u32,
    pub(super) low_gate_s: u32,
    pub(super) low_gate_b: u32,
    pub(super) low_up_s: u32,
    pub(super) low_up_b: u32,
    pub(super) low_down_s: u32,
    pub(super) low_down_b: u32,
}

#[repr(C)]
pub(super) struct QmvParams {
    pub(super) out_dim: u32,
    pub(super) in_dim: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct QkRopeParams {
    pub(super) n_heads: u32,
    pub(super) head_dim: u32,
    pub(super) stride: u32,
    pub(super) rot: u32,
    pub(super) pos: u32,
    pub(super) theta: f32,
    pub(super) eps: f32,
}

#[repr(C)]
pub(super) struct ConvParams {
    pub(super) channels: u32,
    pub(super) ksize: u32,
}

#[repr(C)]
pub(super) struct DeltaPrepParams {
    pub(super) n_k: u32,
    pub(super) n_v: u32,
    pub(super) d_k: u32,
    pub(super) d_v: u32,
    pub(super) eps: f32,
    pub(super) nb: u32,
    pub(super) snap_after: u32,
}

#[repr(C)]
pub(super) struct AttnPartParams {
    pub(super) n_heads: u32,
    pub(super) n_kv: u32,
    pub(super) head_dim: u32,
    pub(super) t_len: u32,
    pub(super) q_stride: u32,
    pub(super) q_off: u32,
    pub(super) max_blk: u32,
    pub(super) scale: f32,
}

#[repr(C)]
pub(super) struct KvQParams {
    pub(super) row: u32,
    pub(super) t0: u32,
    pub(super) nb: u32,
    pub(super) scale_off: u32,
}

#[repr(C)]
pub(super) struct PleConvParams {
    pub(super) channels: u32,
    pub(super) ksize: u32,
    pub(super) dilation: u32,
    pub(super) span: u32,
    pub(super) filled: u32,
    pub(super) nb: u32,
}

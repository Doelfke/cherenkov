//! Router and resident/fetched expert compute dispatch.

use super::*;

impl Gpu<'_> {
    /// Router over `nb` rows of `x`: softmax over experts, top-k.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn router_b(
        &self,
        enc: &Enc,
        moe: &Moe,
        nb: usize,
        x: &Buf,
        logits: &Buf,
        idx: &Buf,
        w: &Buf,
        k: usize,
    ) {
        let c = &self.p.cfg;
        let rows = c.num_experts as u32;
        let cols = c.hidden_size as u32;
        let nbu = nb as u32;
        self.dispatch(
            enc,
            &self.pipes.bf16_matvec_b,
            |e| {
                self.bind(e, 0, &self.dense, moe.router.0);
                self.bind(e, 1, x, 0);
                self.bind(e, 2, logits, 0);
                set_bytes(e, 3, &rows);
                set_bytes(e, 4, &cols);
                set_bytes(e, 5, &nbu);
            },
            (nb * c.num_experts).div_ceil(4),
            128,
            true,
        );
        let k = k as u32;
        let renorm = c.norm_topk_prob as u32;
        self.dispatch(
            enc,
            &self.pipes.topk_softmax_b,
            |e| {
                self.bind(e, 0, logits, 0);
                self.bind(e, 1, idx, 0);
                self.bind(e, 2, w, 0);
                set_bytes(e, 3, &rows);
                set_bytes(e, 4, &k);
                set_bytes(e, 5, &renorm);
            },
            nb,
            c.num_experts.max(32),
            true,
        );
    }

    /// Routed experts of `nb` rows (the union published in slot table row
    /// `slot_row` at execution time) plus the shared expert, summed per
    /// row into scratch.moe_out. Part 0 covers the records resident when
    /// the table was published (plus the shared expert); part 1 the ones
    /// fetched meanwhile. Input: scratch.hc.mixed (prepped in h1).
    pub(super) fn experts_b(&self, enc: &Enc, moe: &Moe, slot_row: usize, nb: usize, part: u32) {
        let c = &self.p.cfg;
        let s = &self.scratch;
        let l = super::super::lowbit::Layout::four_bit(&self.p.manifest.experts);
        let inter = self.p.manifest.experts.inter as u32;
        let h = c.hidden_size as u32;
        let k = c.num_experts_per_tok;
        let shared = !self.skips("shared");
        if self.skips("experts") {
            if part == 0 {
                self.zero(enc, &s.moe_out, nb as u32 * h);
            }
            return;
        }
        let n_max = (k * nb) as u32;
        let mp = MoeBParams {
            inter,
            hidden: h,
            n_u: n_max,
            gate_w: l.gate_w as u32,
            up_w: l.up_w as u32,
            down_w: l.down_w as u32,
            gate_s: l.gate_s as u32,
            gate_b: l.gate_b as u32,
            up_s: l.up_s as u32,
            up_b: l.up_b as u32,
            down_s: l.down_s as u32,
            down_b: l.down_b as u32,
            layer: slot_row as u32,
            shared: shared as u32,
            nb: nb as u32,
            sh_gate_w: moe.sg.w as u32,
            sh_gate_s: moe.sg.s as u32,
            sh_gate_b: moe.sg.b as u32,
            sh_up_w: moe.su.w as u32,
            sh_up_s: moe.su.s as u32,
            sh_up_b: moe.su.b as u32,
            sh_down_w: moe.sd.w as u32,
            sh_down_s: moe.sd.s as u32,
            sh_down_b: moe.sd.b as u32,
            sh_gate_vec: moe.shared_gate.0 as u32,
            part,
            low_gate_w: self.low_bit_store.map_or(0, |s| s.gate_w as u32),
            low_up_w: self.low_bit_store.map_or(0, |s| s.up_w as u32),
            low_down_w: self.low_bit_store.map_or(0, |s| s.down_w as u32),
            low_gate_s: self.low_bit_store.map_or(0, |s| s.gate_s as u32),
            low_gate_b: self.low_bit_store.map_or(0, |s| s.gate_b as u32),
            low_up_s: self.low_bit_store.map_or(0, |s| s.up_s as u32),
            low_up_b: self.low_bit_store.map_or(0, |s| s.up_b as u32),
            low_down_s: self.low_bit_store.map_or(0, |s| s.down_s as u32),
            low_down_b: self.low_bit_store.map_or(0, |s| s.down_b as u32),
        };
        let n_exp = (n_max + 1) as usize;
        let rows2 = 2 * inter as usize;
        self.dispatch(
            enc,
            &self.pipes.moe_gate_up_b[nb - 1],
            |e| {
                self.bind(e, 0, &s.hc.h1.xe, 0);
                self.bind(e, 1, &s.hc.h1.xo, 0);
                self.bind(e, 2, &s.hc.h1.xsum, 0);
                self.bind(e, 3, &s.gate_e, 0);
                set_bytes(e, 4, &mp);
                self.bind(e, 5, &self.slot_tab, 0);
                self.bind(e, 6, &self.dense, 0);
            },
            (n_exp * rows2).div_ceil(4),
            128,
            true,
        );
        self.dispatch(
            enc,
            &self.pipes.moe_act_b,
            |e| {
                self.bind(e, 0, &s.gate_e, 0);
                self.bind(e, 1, &s.hx.xe, 0);
                self.bind(e, 2, &s.hx.xo, 0);
                self.bind(e, 3, &s.hx.xsum, 0);
                set_bytes(e, 4, &mp);
                self.bind(e, 5, &self.slot_tab, 0);
            },
            n_exp * nb * inter as usize / 2,
            256,
            false,
        );
        self.dispatch(
            enc,
            &self.pipes.moe_down_b[nb - 1],
            |e| {
                self.bind(e, 0, &s.hx.xe, 0);
                self.bind(e, 1, &s.hx.xo, 0);
                self.bind(e, 2, &s.hx.xsum, 0);
                self.bind(e, 3, &s.y_e, 0);
                set_bytes(e, 4, &mp);
                self.bind(e, 5, &self.slot_tab, 0);
                self.bind(e, 6, &self.dense, 0);
                // Two output rows per simdgroup.
            },
            (n_exp * (h as usize / 2)).div_ceil(4),
            128,
            true,
        );
        self.dispatch(
            enc,
            &self.pipes.moe_combine_b,
            |e| {
                self.bind(e, 0, &s.y_e, 0);
                self.bind(e, 1, &self.wmap, 0);
                self.bind(e, 2, &s.moe_out, 0);
                set_bytes(e, 3, &mp);
                self.bind(e, 4, &self.dense, 0);
                self.bind(e, 5, &s.hc.mixed, 0);
                self.bind(e, 6, &self.slot_tab, 0);
            },
            nb * (h as usize).div_ceil(256),
            256,
            true,
        );
    }
}

// SPDX-License-Identifier: Apache-2.0
//! Batch bookkeeping: turns vLLM-style per-sequence metadata (query length,
//! context length, block table) into ggml inputs: RoPE positions, KV slot
//! mapping, per-group K/V gather indices and attention masks.
//!
//! Tokens are packed sequence-major. Consecutive sequences with the same query
//! length form a *group*; each group is attended with one `flash_attn_ext`
//! call over `[head_dim, q_len, n_head, n_seqs]` queries against K/V gathered
//! from the paged cache (padded to the longest context in the group).

use anyhow::{bail, Result};

const F16_ZERO: u16 = 0x0000;
const F16_NEG_INF: u16 = 0xFC00;

#[derive(Clone, Debug)]
pub struct Seq {
    pub q_len: usize,
    /// Context length after this step (= tokens already cached + q_len).
    pub ctx_len: usize,
    pub blocks: Vec<i32>,
    pub state_slot: i32,
    pub reset_state: bool,
    /// Compute logits for the last `n_logits` tokens of this sequence.
    pub n_logits: usize,
}

#[derive(Clone, Debug)]
pub struct Group {
    pub seq_start: usize,
    pub n_seqs: usize,
    pub q_len: usize,
    /// First token (within the micro-batch) of the group.
    pub tok_start: usize,
}

/// A self-contained sub-batch executed as one ggml graph.
#[derive(Debug)]
pub struct MicroBatch {
    pub tokens: Vec<i32>,
    pub seqs: Vec<Seq>,
    pub groups: Vec<Group>,
    pub block_size: usize,
}

pub struct AttnMeta {
    /// Number of gathered keys per sequence (padded).
    pub n_kv: usize,
    /// Cache slot per gathered key, `[n_kv * n_seqs]`.
    pub gather: Vec<i32>,
    /// F16 bits, layout `[n_kv, q_len, 1, n_seqs]`.
    pub mask: Vec<u16>,
}

impl MicroBatch {
    pub fn n_tokens(&self) -> usize {
        self.tokens.len()
    }

    fn tok_offsets(&self) -> Vec<usize> {
        let mut out = Vec::with_capacity(self.seqs.len());
        let mut t = 0;
        for s in &self.seqs {
            out.push(t);
            t += s.q_len;
        }
        out
    }

    fn slot(&self, s: &Seq, pos: usize) -> Result<i64> {
        let b = pos / self.block_size;
        if b >= s.blocks.len() {
            bail!(
                "position {pos} beyond block table ({} blocks)",
                s.blocks.len()
            );
        }
        Ok(s.blocks[b] as i64 * self.block_size as i64 + (pos % self.block_size) as i64)
    }

    pub fn positions(&self) -> Vec<i32> {
        let mut out = Vec::with_capacity(self.n_tokens());
        for s in &self.seqs {
            let p0 = s.ctx_len - s.q_len;
            out.extend((0..s.q_len).map(|j| (p0 + j) as i32));
        }
        out
    }

    pub fn slot_mapping(&self) -> Result<Vec<i64>> {
        let mut out = Vec::with_capacity(self.n_tokens());
        for s in &self.seqs {
            let p0 = s.ctx_len - s.q_len;
            for j in 0..s.q_len {
                out.push(self.slot(s, p0 + j)?);
            }
        }
        Ok(out)
    }

    /// Token indices (within the micro-batch) whose logits are requested.
    pub fn logit_rows(&self) -> Vec<i32> {
        let offs = self.tok_offsets();
        let mut out = Vec::new();
        for (s, &t0) in self.seqs.iter().zip(&offs) {
            let n = s.n_logits.min(s.q_len);
            out.extend((s.q_len - n..s.q_len).map(|j| (t0 + j) as i32));
        }
        out
    }

    /// Gather indices + mask for `group`. `window` is the sliding window
    /// size (keys with `pos > q_pos - window` are visible), `None` = causal.
    pub fn attn_meta(&self, group: &Group, window: Option<usize>) -> Result<AttnMeta> {
        let seqs = &self.seqs[group.seq_start..group.seq_start + group.n_seqs];
        let q = group.q_len;
        let starts: Vec<usize> = seqs
            .iter()
            .map(|s| {
                let p0 = s.ctx_len - s.q_len;
                match window {
                    Some(w) => (p0 + 1).saturating_sub(w),
                    None => 0,
                }
            })
            .collect();
        let n_kv = seqs
            .iter()
            .zip(&starts)
            .map(|(s, &st)| s.ctx_len - st)
            .max()
            .unwrap_or(1)
            .max(1);

        let mut gather = vec![0i32; n_kv * seqs.len()];
        let mut mask = vec![F16_NEG_INF; n_kv * q * seqs.len()];
        for (si, (s, &st)) in seqs.iter().zip(&starts).enumerate() {
            let n = s.ctx_len - st;
            for i in 0..n {
                gather[si * n_kv + i] = self.slot(s, st + i)? as i32;
            }
            let p0 = s.ctx_len - s.q_len;
            for j in 0..q {
                let qp = p0 + j;
                let row = &mut mask[(si * q + j) * n_kv..(si * q + j + 1) * n_kv];
                for (i, m) in row.iter_mut().enumerate().take(n) {
                    let kp = st + i;
                    let visible = kp <= qp && window.is_none_or(|w| kp + w > qp);
                    if visible {
                        *m = F16_ZERO;
                    }
                }
            }
        }
        Ok(AttnMeta { n_kv, gather, mask })
    }
}

/// Limits applied when splitting a step into micro-batches.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub max_groups: usize,
    /// Byte budget for per-step attention temporaries (gathered F32 K/V and
    /// F16 masks), summed over groups.
    pub attn_bytes: usize,
    /// Largest `head_dim * n_kv_heads` over KV layers.
    pub kv_row: usize,
    /// Number of attention kinds whose gathered K/V can be alive at once
    /// (e.g. causal + sliding for models with cross-layer KV sharing).
    pub kinds: usize,
}

impl Limits {
    #[cfg(test)]
    pub fn unlimited(max_groups: usize) -> Self {
        Self {
            max_groups,
            attn_bytes: usize::MAX,
            kv_row: 0,
            kinds: 1,
        }
    }

    /// Temporary bytes for a group of `n` sequences, `q` query tokens each,
    /// attending over up to `ctx` keys.
    fn group_bytes(&self, ctx: usize, q: usize, n: usize) -> usize {
        // K + V gathered as F32, plus the F16 mask.
        let per_key = 2 * 4 * self.kv_row + 2 * q;
        ctx.saturating_mul(n)
            .saturating_mul(per_key)
            .saturating_mul(self.kinds)
    }
}

/// Split a full batch into micro-batches honoring `limits`.
pub fn plan(
    tokens: &[i32],
    seqs: Vec<Seq>,
    block_size: usize,
    limits: Limits,
) -> Result<Vec<MicroBatch>> {
    let total: usize = seqs.iter().map(|s| s.q_len).sum();
    if total != tokens.len() {
        bail!(
            "sum of q_lens ({total}) != number of tokens ({})",
            tokens.len()
        );
    }
    for s in &seqs {
        if s.q_len == 0 || s.ctx_len < s.q_len {
            bail!("invalid sequence: q_len={} ctx_len={}", s.q_len, s.ctx_len);
        }
    }

    let mut out = Vec::new();
    let mut cur_tokens = Vec::new();
    let mut cur_seqs: Vec<Seq> = Vec::new();
    let mut cur_groups: Vec<Group> = Vec::new();
    // Max context of each current group, and the running temporary bytes.
    let mut group_ctx: Vec<usize> = Vec::new();
    let mut load = 0usize;
    let mut t = 0usize;
    for s in seqs {
        let continues = cur_groups.last().is_some_and(|g| g.q_len == s.q_len);
        let new_load = if continues {
            let g = cur_groups.last().unwrap();
            let c = *group_ctx.last().unwrap();
            load - limits.group_bytes(c, g.q_len, g.n_seqs)
                + limits.group_bytes(c.max(s.ctx_len), g.q_len, g.n_seqs + 1)
        } else {
            load + limits.group_bytes(s.ctx_len, s.q_len, 1)
        };
        let over_groups = !continues && cur_groups.len() == limits.max_groups;
        let over_bytes = new_load > limits.attn_bytes && !cur_seqs.is_empty();
        if over_groups || over_bytes {
            out.push(MicroBatch {
                tokens: std::mem::take(&mut cur_tokens),
                seqs: std::mem::take(&mut cur_seqs),
                groups: std::mem::take(&mut cur_groups),
                block_size,
            });
            group_ctx.clear();
            load = 0;
        }
        let continues = cur_groups.last().is_some_and(|g| g.q_len == s.q_len);
        if continues {
            let g = cur_groups.last_mut().unwrap();
            let c = group_ctx.last_mut().unwrap();
            load -= limits.group_bytes(*c, g.q_len, g.n_seqs);
            g.n_seqs += 1;
            *c = (*c).max(s.ctx_len);
            load += limits.group_bytes(*c, g.q_len, g.n_seqs);
        } else {
            group_ctx.push(s.ctx_len);
            load += limits.group_bytes(s.ctx_len, s.q_len, 1);
            cur_groups.push(Group {
                seq_start: cur_seqs.len(),
                n_seqs: 1,
                q_len: s.q_len,
                tok_start: cur_tokens.len(),
            });
        }
        cur_tokens.extend_from_slice(&tokens[t..t + s.q_len]);
        t += s.q_len;
        cur_seqs.push(s);
    }
    if !cur_seqs.is_empty() {
        out.push(MicroBatch {
            tokens: cur_tokens,
            seqs: cur_seqs,
            groups: cur_groups,
            block_size,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seq(q: usize, c: usize) -> Seq {
        Seq {
            q_len: q,
            ctx_len: c,
            blocks: vec![3, 1, 2],
            state_slot: 0,
            reset_state: false,
            n_logits: 1,
        }
    }

    #[test]
    fn groups_split_on_qlen() {
        let toks = vec![0; 1 + 1 + 4 + 2];
        let mbs = plan(
            &toks,
            vec![seq(1, 5), seq(1, 9), seq(4, 4), seq(2, 6)],
            4,
            Limits::unlimited(8),
        )
        .unwrap();
        assert_eq!(mbs.len(), 1);
        let g = &mbs[0].groups;
        assert_eq!(g.len(), 3);
        assert_eq!((g[0].n_seqs, g[0].q_len, g[0].tok_start), (2, 1, 0));
        assert_eq!((g[1].n_seqs, g[1].q_len, g[1].tok_start), (1, 4, 2));
        assert_eq!((g[2].n_seqs, g[2].q_len, g[2].tok_start), (1, 2, 6));
        assert_eq!(mbs[0].logit_rows(), vec![0, 1, 5, 7]);
    }

    #[test]
    fn micro_batches_respect_group_limit() {
        let toks = vec![0; 2 + 3 + 4];
        let mbs = plan(
            &toks,
            vec![seq(2, 2), seq(3, 3), seq(4, 4)],
            4,
            Limits::unlimited(2),
        )
        .unwrap();
        assert_eq!(mbs.len(), 2);
        assert_eq!(mbs[0].tokens.len(), 5);
        assert_eq!(mbs[1].groups[0].tok_start, 0);
    }

    #[test]
    fn micro_batches_respect_attention_byte_budget() {
        // 4 decode seqs with ctx 100: each costs 100 * (2*4*8 + 2) = 6600 bytes.
        let limits = Limits {
            max_groups: 8,
            attn_bytes: 2 * 6600,
            kv_row: 8,
            kinds: 1,
        };
        let toks = vec![0; 4];
        let seqs = (0..4).map(|_| seq(1, 100)).collect();
        let mbs = plan(&toks, seqs, 4, limits).unwrap();
        assert_eq!(
            mbs.iter().map(|m| m.seqs.len()).collect::<Vec<_>>(),
            vec![2, 2]
        );
        // A single sequence larger than the budget still runs on its own.
        let big = plan(&[0; 1], vec![seq(1, 10_000)], 4, limits).unwrap();
        assert_eq!(big.len(), 1);
    }

    #[test]
    fn slots_and_mask() {
        let mb = &plan(&[0; 2], vec![seq(2, 6)], 4, Limits::unlimited(8)).unwrap()[0];
        // positions 4,5 -> block index 1 -> physical block 1
        assert_eq!(mb.positions(), vec![4, 5]);
        assert_eq!(mb.slot_mapping().unwrap(), vec![4, 5]);
        let m = mb.attn_meta(&mb.groups[0], None).unwrap();
        assert_eq!(m.n_kv, 6);
        assert_eq!(&m.gather[..], &[12, 13, 14, 15, 4, 5]);
        // query at pos 4 sees keys 0..=4
        assert_eq!(&m.mask[..6], &[0, 0, 0, 0, 0, F16_NEG_INF]);
        assert_eq!(&m.mask[6..], &[0; 6]);
        // window 2: query 4 sees 3,4 ; query 5 sees 4,5 ; gather starts at 3
        let w = mb.attn_meta(&mb.groups[0], Some(2)).unwrap();
        assert_eq!(w.n_kv, 3);
        assert_eq!(&w.gather[..], &[15, 4, 5]);
        assert_eq!(&w.mask[..], &[0, 0, F16_NEG_INF, F16_NEG_INF, 0, 0]);
    }
}

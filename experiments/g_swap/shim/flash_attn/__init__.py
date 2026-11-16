"""Minimal pure-PyTorch stand-in for the two flash_attn entry points Infinity imports.
Only flash_attn_varlen_kvpacked_func is exercised at inference (cross-attention)."""
import torch
import torch.nn.functional as F

def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, **kw):
    # q,k,v: B L H c  -> B L H c
    o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                       scale=softmax_scale, is_causal=causal)
    return o.transpose(1, 2)

def flash_attn_varlen_kvpacked_func(q, kv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                                    dropout_p=0.0, softmax_scale=None, causal=False, **kw):
    # q: Nq H c ; kv: Nk 2 H c ; returns Nq H c
    out = torch.empty_like(q)
    cq, ck = cu_seqlens_q.tolist(), cu_seqlens_k.tolist()
    for i in range(len(cq) - 1):
        qi = q[cq[i]:cq[i + 1]].transpose(0, 1).unsqueeze(0)            # 1 H Lq c
        ki = kv[ck[i]:ck[i + 1], 0].transpose(0, 1).unsqueeze(0)
        vi = kv[ck[i]:ck[i + 1], 1].transpose(0, 1).unsqueeze(0)
        o = F.scaled_dot_product_attention(qi, ki, vi, scale=softmax_scale)
        out[cq[i]:cq[i + 1]] = o[0].transpose(0, 1)
    return out

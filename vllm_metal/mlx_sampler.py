import mlx.core as mx

def mlx_sample(logits: mx.array, temp: float = 0.0, top_p: float = 1.0, top_k: int = -1, min_p: float = 0.0) -> mx.array:
    if temp == 0.0:
        return mx.argmax(logits, axis=-1)

    logits = logits / temp
    
    if min_p > 0.0:
        probs = mx.softmax(logits, axis=-1)
        top_prob = mx.max(probs, axis=-1, keepdims=True)
        scaled_min_p = top_prob * min_p
        mask = probs < scaled_min_p
        logits = mx.where(mask, -float('inf'), logits)

    if top_k > 0:
        top_k_indices = mx.argpartition(logits, -top_k, axis=-1)[:, -top_k:]
        top_k_values = mx.take_along_axis(logits, top_k_indices, axis=-1)
        min_top_k = mx.min(top_k_values, axis=-1, keepdims=True)
        
        mask = logits < min_top_k
        logits = mx.where(mask, -float('inf'), logits)

    if top_p < 1.0 and top_p > 0.0:
        sorted_indices = mx.argsort(logits, axis=-1)[..., ::-1]
        sorted_logits = mx.take_along_axis(logits, sorted_indices, axis=-1)
        
        sorted_probs = mx.softmax(sorted_logits, axis=-1)
        cumulative_probs = mx.cumsum(sorted_probs, axis=-1)
        
        mask = cumulative_probs > top_p
        mask = mx.concatenate([mx.zeros((mask.shape[0], 1), dtype=bool), mask[..., :-1]], axis=-1)
        
        sorted_logits = mx.where(mask, -float('inf'), sorted_logits)
        sorted_sample_idx = mx.random.categorical(sorted_logits)
        
        expanded_idx = sorted_sample_idx[..., None]
        original_token_id = mx.take_along_axis(sorted_indices, expanded_idx, axis=-1)
        return original_token_id.squeeze(-1)

    return mx.random.categorical(logits)

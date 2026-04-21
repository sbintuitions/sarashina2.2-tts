import torch
import re
from typing import List, Optional


def audio_decode_flash(
    audio_token_list: list,
    codec_decoder,   # FlowDecoder instance
    *,
    flow_embedding_only: bool = False,
    code_layer: int = 1,
    num_latency_tokens: int = 1,
    speed: float = 1.0,
    flow_embedding: Optional[torch.Tensor] = None,
    prompt_tokens: Optional[list] = None,
    prompt_feat: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Decode audio tokens → waveform via FlowDecoder (single utterance).

    Parameters
    ----------
    audio_token_list : list of tensors — token tensors for one utterance
    codec_decoder : FlowDecoder instance
    flow_embedding : (1, 192) speaker embedding
    prompt_tokens : list of int — prompt speech token ids
    prompt_feat : (1, T, 80) prompt mel features

    Returns
    -------
    (1, wave_len) waveform tensor
    """
    if not audio_token_list:
        return torch.zeros(1, 0)

    # Prepare token tensor
    if code_layer > 1:
        t = torch.stack(audio_token_list, dim=0).permute(1, 0).reshape(-1)
        t = t[num_latency_tokens * code_layer:]
    else:
        t = torch.cat(audio_token_list, dim=-1)
        t = t[num_latency_tokens:]

    # Prepare prompt data
    if flow_embedding_only:
        emb = torch.zeros(192) if flow_embedding is None else flow_embedding.flatten()[:192]
        ptok = torch.zeros(0, dtype=torch.int32)
        pfeat = torch.zeros(0, 80)
    else:
        emb = flow_embedding.flatten()[:192] if flow_embedding is not None else torch.zeros(192)
        ptok = torch.tensor(prompt_tokens, dtype=torch.int32) if prompt_tokens else torch.zeros(0, dtype=torch.int32)
        pfeat = prompt_feat.squeeze(0) if prompt_feat is not None else torch.zeros(0, 80)

    return codec_decoder.token2wav(
        token=t,
        prompt_token=ptok,
        prompt_feat=pfeat,
        embedding=emb,
        speed=speed,
    )


def audio_tokens_to_tensor_list(token_string):
    """
    Convert a string of audio semantic tokens into a list of tensors.
    
    Args:
        token_string (str): The input string containing semantic tokens. 
        e.g., "<|semantic_0|><|semantic_1|>...<|semantic_n|>"
    Returns:
        list: A list of tensors representing the semantic tokens.
        e.g., [tensor([0]), tensor([1]), ..., tensor([n])]
    """
    numbers = re.findall(r'<\|semantic_(\d+)\|>', token_string)
    if not numbers:
        print("⚠️ No valid semantic tokens found in input string.")
        return []
    try:
        numbers = [int(num) for num in numbers]
    except ValueError as e:
        print(f"⚠️ Error parsing numbers: {e}")
        return []
    tensor_list = [torch.tensor([num]) for num in numbers]
    return tensor_list

def audio_tokens_to_tensor_list_batch(token_batch):
    """
    Convert a batch of audio semantic tokens strings into a list of lists of tensors.
    
    Args:
        token_batch (list of str): A list of input strings containing semantic tokens. 
        e.g., ["<|semantic_0|><|semantic_1|>", "<|semantic_2|>...<|semantic_n|>"]
    Returns:
        list of list: A list where each element is a list of tensors representing the semantic tokens.
        e.g., [[tensor([0]), tensor([1])], [tensor([2]), ..., tensor([n])]]
    """
    result = []
    for token_string in token_batch:
        tensor_list = audio_tokens_to_tensor_list(token_string)
        result.append(tensor_list)
    return result

'''
For managing additional tokens. 

This module provides functionality to handle additional tokens, 
including additional special tokens and speech tokens. 
'''
S3_VOCAB_SIZE = 6561    # speech_tokenizer_v2_25hz has a codebook size of 6561. V1 has 4096.

SPEECH_START_TOKEN = "<|speech_start|>" # Only using this for indicate speech turn. 
SPEECH_END_TOKEN = "<|speech_end|>"     # Not using this. 
PRON_START_TOKEN = "<|pron_start|>"
PRON_END_TOKEN = "<|pron_end|>"
PAD_TOKEN = "<|pad|>"

RESEARVED_TOKEN_TEMPLATE = "<|reserved_{}|>"
RESEARVED_TOKENS = [
    RESEARVED_TOKEN_TEMPLATE.format(i) for i in range(20)
]

SEMANTIC_TOKEN_TEMPLATE = "<|semantic_{}|>"
SEMANTIC_TOKENS = [
    SEMANTIC_TOKEN_TEMPLATE.format(i) for i in range(S3_VOCAB_SIZE)
]

# When adding new tokens, ensure they are added to the end of the list.
# The tokenizer will first add SEMANTIC_TOKENS, then special tokens.
ALL_SPECIAL_TOKENS = [
    SPEECH_START_TOKEN,
    SPEECH_END_TOKEN,
    PRON_START_TOKEN,
    PRON_END_TOKEN,
    PAD_TOKEN,
    *RESEARVED_TOKENS,
]
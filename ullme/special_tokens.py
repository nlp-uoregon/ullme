SPECIAL_TOKENS = {
    'xlm-r': {
        'bos': '<s>',
        'eos': '</s>',
    },
    'mistral': {
        'bos': '<s>',
        'eos': '</s>',
        'pad': '</s>',
        'mask': '<unk>',
    },
    'llama': {
        'bos': '<|begin_of_text|>',
        'eos': '<|end_of_text|>',
        'pad': '<|end_of_text|>',
        'mask': "<|reserved_special_token_0|>",
    },
    'nvidia/NV-Embed-v2': {
        'bos': '<s>',
        'eos': '</s>',
        'pad': '</s>',
        'mask': '<unk>',
    },
    'phi': {
        'bos': '<|endoftext|>',
        'eos': '<|endoftext|>',
        'pad': '<|endoftext|>',
        'mask': '<|endoftext|>',
    }
}
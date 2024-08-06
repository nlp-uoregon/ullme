SPECIAL_TOKENS = {
    't5': {
        'eos': '</s>',
    },
    'llama': {
        'bos': '<|begin_of_text|>',
        'user_bos': '<|start_header_id|>user<|end_header_id|>\n\n',
        'eot': '<|eot_id|>\n',
        'assistant_bos': '<|start_header_id|>assistant<|end_header_id|>\n\n',
        'eos': '<|end_of_text|>'
    },
    'mistral':{
        'bos': '<s>',
        'user_bos': '[INST]',
        'eot': '',
        'assistant_bos': '[/INST]',
        'eos': '</s>'
    },
    'phi3': {
        'bos': '<s>',
        'user_bos': '<|user|>',
        'eot': '<|end|>',
        'assistant_bos': '<|assistant|>',
        'eos': '<|end_of_text|>'
    },
    'phi':{
        'bos': '',
        'user_bos': '<|im_start|>user\n\n',
        'eot': '\n',
        'assistant_bos': '<|im_start|>assistant\n\n',
        'eos': '<|im_end|>'
    },
    'cohere': {
        'bos': '<BOS_TOKEN>',
        'user_bos': '<|START_OF_TURN_TOKEN|><|USER_TOKEN|>',
        'eot': '<|END_OF_TURN_TOKEN|>',
        'assistant_bos': '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>',
        'eos': '<|END_OF_TURN_TOKEN|>'
    },
    'gemma': {
        'eos': '</s>',
    },
    'gemma2': {
        'eos': '</s>',
    },

}
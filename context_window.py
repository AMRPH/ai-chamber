"""Fit chat history into the input budget while preserving complete turns."""


def fit_chat_context(tokenizer, messages, max_input_tokens, return_tensors=None):
    kept = list(messages)
    prefix = 0
    while prefix < len(kept) and kept[prefix]['role'] == 'system':
        prefix += 1
    dropped = 0
    while True:
        encoded = tokenizer.apply_chat_template(
            kept, tokenize=True, add_generation_prompt=True, enable_thinking=False,
            return_dict=return_tensors is not None,
            **({'return_tensors': return_tensors} if return_tensors is not None else {}))
        length = encoded['input_ids'].shape[-1] if return_tensors is not None else len(encoded)
        if length <= max_input_tokens:
            return encoded, dropped
        if len(kept) <= prefix + 1:
            raise ValueError('Сообщение слишком длинное для контекста модели.')
        # Drop the oldest user/assistant turn, leaving the final user message.
        end = prefix + 1
        while end < len(kept) - 1 and kept[end]['role'] != 'user':
            end += 1
        dropped += end - prefix
        del kept[prefix:end]

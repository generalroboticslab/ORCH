DEFAULT_CONTEXT_LIMITS = {
    "gpt": 256_000,
    "qwen": 256_000,
    "deepseek": 256_000,
    "gemma": 256_000,
    "glm": 180_000,
    "llama": 256_000,
    "ernie": 128_000,
    "nemotron": 128_000,
    "minimax": 128_000,
    "kimi": 256_000,
}
DEFAULT_CONTEXT_LIMIT = 128_000
RESERVED_OUTPUT_TOKENS = 4_096
AUTO_TRUNCATE = True
PRINT_TOKEN_COUNTS = False


def is_context_length_bad_request(error):
    message = str(error).lower()
    return (
        "maximum context length" in message
        or "reduce the length of the input prompt" in message
        or "input_tokens" in message
    )


def context_limit_for_model(model):
    model_lower = (model or "").lower()
    for key, limit in DEFAULT_CONTEXT_LIMITS.items():
        if key in model_lower:
            return limit
    return DEFAULT_CONTEXT_LIMIT


def reserved_output_tokens():
    return RESERVED_OUTPUT_TOKENS


def input_token_budget(model):
    return max(1, context_limit_for_model(model) - reserved_output_tokens())


def count_message_tokens(messages, model=None):
    try:
        import tiktoken

        try:
            encoding = tiktoken.encoding_for_model(model or "gpt-4o")
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        total = 0
        for message in messages:
            total += 4
            total += len(encoding.encode(str(message.get("role", ""))))
            total += len(encoding.encode(str(message.get("content", ""))))
        return total + 2
    except Exception:
        chars = sum(
            len(str(message.get("role", ""))) + len(str(message.get("content", "")))
            for message in messages
        )
        return max(1, chars // 3)


class ContextLengthPreflightError(ValueError):
    pass


def auto_truncate_enabled():
    return AUTO_TRUNCATE


def print_token_counts_enabled():
    return PRINT_TOKEN_COUNTS


def _content_token_count(content, model=None):
    return count_message_tokens([{"role": "user", "content": content}], model)


def _truncate_text_to_token_budget(text, budget, model=None, label="Wildfire"):
    text = str(text)
    if budget <= 0:
        return ""
    if _content_token_count(text, model) <= budget:
        return text

    marker = f"\n\n[... {label} prompt truncated to fit context window ...]\n\n"
    low = 0
    high = len(text)
    best = ""
    while low <= high:
        keep = (low + high) // 2
        head_len = keep // 2
        tail_len = keep - head_len
        candidate = text[:head_len] + marker + (text[-tail_len:] if tail_len else "")
        if _content_token_count(candidate, model) <= budget:
            best = candidate
            low = keep + 1
        else:
            high = keep - 1
    return best or marker.strip()


def shorten_messages_to_fit(model, messages, label="Wildfire"):
    shortened = [dict(message) for message in messages]
    budget = input_token_budget(model)
    original_count = count_message_tokens(shortened, model)
    token_count = original_count
    if token_count <= budget:
        return shortened, token_count, False

    if not auto_truncate_enabled():
        return shortened, token_count, False

    for _ in range(12):
        largest_index = max(
            range(len(shortened)),
            key=lambda i: len(str(shortened[i].get("content", ""))),
            default=0,
        )
        content = str(shortened[largest_index].get("content", ""))
        other_messages = [
            message for i, message in enumerate(shortened) if i != largest_index
        ]
        other_tokens = count_message_tokens(other_messages, model) if other_messages else 0
        content_budget = max(1, budget - other_tokens - 16)
        shortened[largest_index]["content"] = _truncate_text_to_token_budget(
            content,
            content_budget,
            model,
            label=label,
        )
        token_count = count_message_tokens(shortened, model)
        if token_count <= budget:
            if print_token_counts_enabled():
                print(f"{label}: auto-truncated prompt from {original_count} to {token_count}/{budget} tokens")
            return shortened, token_count, True

    return shortened, token_count, True


def check_messages_fit(model, messages, label="Wildfire"):
    _, token_count, _ = shorten_messages_to_fit(model, messages, label)
    budget = input_token_budget(model)
    if token_count > budget:
        raise ContextLengthPreflightError(
            f"{label} input is too long before API call: {token_count} tokens > "
            f"{budget} token input budget for model '{model}'. "
            f"Increase DEFAULT_CONTEXT_LIMITS/RESERVED_OUTPUT_TOKENS in token_guard.py "
            f"if this hardcoded limit is wrong."
        )
    if print_token_counts_enabled():
        print(f"{label}: {token_count}/{budget} input tokens for {model}")
    return token_count


def guarded_chat_completion(
    client,
    *,
    model,
    messages,
    label="Wildfire",
    return_none_on_context_error=False,
    **kwargs,
):
    messages, token_count, _ = shorten_messages_to_fit(model, messages, label)
    budget = input_token_budget(model)
    if token_count > budget:
        raise ContextLengthPreflightError(
            f"{label} input is too long before API call: {token_count} tokens > "
            f"{budget} token input budget for model '{model}'. "
            f"Increase DEFAULT_CONTEXT_LIMITS/RESERVED_OUTPUT_TOKENS in token_guard.py "
            f"if this hardcoded limit is wrong."
        )
    if print_token_counts_enabled():
        print(f"{label}: {token_count}/{budget} input tokens for {model}")
    try:
        return client.chat.completions.create(model=model, messages=messages, **kwargs)
    except Exception as error:
        if return_none_on_context_error and is_context_length_bad_request(error):
            print(f"{label}: context length exceeded during chat completion; returning None")
            return None
        raise

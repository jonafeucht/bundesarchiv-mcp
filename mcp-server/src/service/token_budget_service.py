import tiktoken

enc = tiktoken.get_encoding("o200k_base")


def count_tokens(text: str) -> int:
    return len(enc.encode(text))


def apply_token_budget(items: list[str], max_tokens: int) -> list[str]:
    used = 0
    kept = []

    for item in items:
        t = count_tokens(item)

        if used + t > max_tokens:
            break

        kept.append(item)
        used += t

    return kept

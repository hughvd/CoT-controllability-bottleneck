"""OLMo native think-block parsing shared by generation and grading."""
def split_reasoning(raw):
    text=raw.removesuffix('<|endoftext|>')
    if text.startswith('<think>'):text=text[len('<think>'):]
    if '</think>' in text:
        reasoning,answer=text.split('</think>',1)
        return reasoning.strip(),answer.strip()
    return text.strip(),''

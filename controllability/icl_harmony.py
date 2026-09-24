"""Render existing ICL messages in GPT-OSS's native Harmony format."""
import harmony_backend as h

def build_tokens(messages, reasoning_effort='high'):
    from openai_harmony import Conversation, Message, Role, SystemContent, ReasoningEffort
    efforts={'low':ReasoningEffort.LOW,'medium':ReasoningEffort.MEDIUM,'high':ReasoningEffort.HIGH}
    rendered=[Message.from_role_and_content(Role.SYSTEM,
              SystemContent.new().with_reasoning_effort(efforts[reasoning_effort]))]
    for msg in messages:
        if msg['role'] not in ('user','assistant'): raise ValueError('Unexpected ICL message role')
        role=Role.USER if msg['role']=='user' else Role.ASSISTANT
        native=Message.from_role_and_content(role,msg['content'])
        # V4 demonstrates reasoning as a visible prior answer, not prior analysis.
        if role==Role.ASSISTANT: native=native.with_channel('final')
        rendered.append(native)
    return list(h._enc().render_conversation_for_completion(Conversation.from_messages(rendered),Role.ASSISTANT))

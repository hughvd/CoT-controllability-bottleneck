"""Regression coverage for parsing seeded and unseeded Harmony completions."""
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import harmony_backend as h


class HarmonyTests(unittest.TestCase):
    def test_unseeded_analysis_header(self):
        raw = '<|channel|>analysis<|message|>We need to reason.<|end|><|start|>assistant<|channel|>final<|message|>ANSWER: C<|return|>'
        self.assertEqual(h.split_channels(raw), ('We need to reason.', 'ANSWER: C'))

    def test_seed_continuation_and_direct_final(self):
        self.assertEqual(h.split_channels('More reasoning.<|end|><|start|>assistant<|channel|>final<|message|>ANSWER: A'), ('More reasoning.', 'ANSWER: A'))
        self.assertEqual(h.split_channels('<|channel|>final<|message|>ANSWER: B'), ('', 'ANSWER: B'))
        self.assertEqual(h.split_channels('Unfinished reasoning'), ('Unfinished reasoning', ''))

    def test_complete_assistant_header(self):
        self.assertEqual(h.split_channels('<|start|>assistant<|channel|>analysis<|message|>Reasoning.<|end|>'), ('Reasoning.', ''))

    def test_no_instruction_omits_developer_message(self):
        class Content:
            @classmethod
            def new(cls): return cls()
            def with_reasoning_effort(self, effort): return self
            def with_instructions(self, instruction): return instruction
        fake = SimpleNamespace(
            Conversation=SimpleNamespace(from_messages=lambda messages: messages),
            Message=SimpleNamespace(from_role_and_content=lambda role, content: (role, content)),
            Role=SimpleNamespace(SYSTEM='system', DEVELOPER='developer', USER='user', ASSISTANT='assistant'),
            SystemContent=Content, DeveloperContent=Content,
            ReasoningEffort=SimpleNamespace(LOW='low', MEDIUM='medium', HIGH='high'))
        encoder = SimpleNamespace(render_conversation_for_completion=lambda messages, role: messages)
        with patch.dict(sys.modules, openai_harmony=fake), patch.object(h, '_enc', return_value=encoder):
            for instruction in ['', 'none', None]:
                self.assertEqual([r for r, _ in h.build_prefill_tokens('Question', [], instruction, '')], ['system', 'user'])
            self.assertEqual([r for r, _ in h.build_prefill_tokens('Question', [], 'USE UPPERCASE', '')], ['system', 'developer', 'user'])


if __name__ == '__main__':
    unittest.main()

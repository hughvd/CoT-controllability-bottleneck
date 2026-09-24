"""Paper control taxonomy; excludes Chinese and character/persona styles."""
GROUPS = {
    'Addition': ['prefix_sentences', 'numbered_sentences', 'suffix_sentences'],
    'Suppression': ['omit_word', 'no_first_person'],
    'Style': ['uppercase_thinking', 'leetspeak', 'spanish'],
}
CONTROLS = sum(GROUPS.values(), [])
LABELS = {
    'prefix_sentences': 'Sentence prefixes', 'numbered_sentences': 'Numbered sentences',
    'suffix_sentences': 'Sentence endings', 'omit_word': 'Omit word',
    'no_first_person': 'No first person', 'uppercase_thinking': 'Uppercase',
    'leetspeak': 'Leetspeak', 'spanish': 'Spanish',
}

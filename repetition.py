"""Detect three consecutive copies of a text block in a streaming reply."""
import re

# Words, numbers and punctuation; Chinese characters do not need word spaces.
_UNITS = re.compile(r'[\u3400-\u9fff\uf900-\ufaff]|[^\W\d_\u3400-\u9fff\uf900-\ufaff]+|\d+|[^\w\s]|_+')


class RepetitionGuard:
    def __init__(self):
        self.previous = []

    def check(self, text, final=False):
        """Return the end of the third copy, or None. Ignore whitespace changes."""
        matches = list(_UNITS.finditer(text))
        if matches and not final and matches[-1].end() == len(text):
            last = matches[-1].group()
            # A partial streamed word could still become a different word.
            if last[-1].isalnum() and not ('\u3400' <= last[-1] <= '\u9fff' or '\uf900' <= last[-1] <= '\ufaff'):
                matches.pop()
        units = [match.group() for match in matches]
        common = 0
        for old, new in zip(self.previous, units):
            if old != new:
                break
            common += 1
        self.previous = units
        for end in range(max(3, common + 1), len(units) + 1):
            for width in range(1, end // 3 + 1):
                if not (units[end - 1] == units[end - width - 1] == units[end - 2 * width - 1]):
                    continue
                block = units[end - width:end]
                if (block == units[end - 2 * width:end - width]
                        and block == units[end - 3 * width:end - 2 * width]
                        and any(unit.isalnum() for unit in block)):
                    return matches[end - 1].end()
        return None

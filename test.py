import re

# Example list of titles
title = 'advanced nanoparticle generation and excitation by lasers in liquids'
#patterns
parts = [
    r'(?=.*\blaser\w*\b)(?=.*\bgener\w*|.*\bsynth\w*)(?=.*\bnano\w*\b|.*\bcolloid\w*)',
    r'(?=.*\bnanop\w*)(?=.*(ablat\w*|fragment\w*))'
]
pattern = r'|'.join(parts)

# Filter titles that match the pattern
print(re.search(pattern, title, re.IGNORECASE))
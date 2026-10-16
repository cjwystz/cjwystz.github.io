"""Project website CV data into RenderCV's export schema."""
from pathlib import Path
import yaml

root = Path(__file__).resolve().parents[1]
source = yaml.safe_load((root / '_data/cv.yml').read_text())['cv']
cv = {key: source[key] for key in ('name', 'email', 'location', 'social_networks')}
cv['headline'] = source['label']
sections = {'Profile': [source['summary']]}
for name, entries in source['sections'].items():
    converted = []
    for entry in entries:
        if name in ('Education', 'Experience'):
            keys = ('institution', 'area', 'company', 'position', 'location', 'start_date', 'end_date', 'summary', 'highlights')
            item = {key: entry[key] for key in keys if key in entry}
            if name == 'Education':
                item['degree'] = entry['studyType']
                item['highlights'] = [entry['score'], entry['courses'], *entry.get('highlights', [])]
        elif name == 'Publications':
            item = {key: entry[key] for key in ('title', 'authors', 'summary', 'url') if key in entry}
            item['journal'] = entry['publisher']
            item['date'] = str(entry['releaseDate'])
        elif name == 'Awards':
            item = {'label': entry['title'], 'details': f"{entry['awarder']} ({entry['date']})"}
        else:
            item = {'label': entry['name'], 'details': entry.get('keywords', entry.get('summary', ''))}
        converted.append(item)
    sections[name] = converted
cv['sections'] = sections
(root / '_data/rendercv-input.yaml').write_text(yaml.safe_dump({'cv': cv}, allow_unicode=True, sort_keys=False))

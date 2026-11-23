"""Render the shared MUB messages with the checkpoint's own conversation class."""
def render_messages(history, template, *, tiles, image_tokens):
    if tiles < 1 or image_tokens < 1:
        raise ValueError('Missing InternVL image features')
    turns = list(history)
    if turns and turns[0]['role'] == 'system':
        template.system_message = turns.pop(0)['content']
    if [m['role'] for m in turns] not in (['user'], ['user', 'assistant', 'user']):
        raise ValueError('Use the original question and its real answer before feedback')
    count = 0
    for message in turns:
        content = message['content']
        if isinstance(content, list):
            parts = []
            for part in content:
                if part['type'] == 'image':
                    count += 1
                    parts.append('<image>\n')
                elif part['type'] == 'text':
                    parts.append(part['text'])
                else:
                    raise ValueError('Unsupported InternVL content type')
            content = ''.join(parts)
        role = template.roles[0 if message['role'] == 'user' else 1]
        template.append_message(role, content)
    if count != 1:
        raise ValueError('Exactly one original image is required throughout the dialogue')
    template.append_message(template.roles[1], None)
    text = template.get_prompt()
    if text.count('<image>') != 1:
        raise ValueError('Expected one image placeholder')
    replacement = '<img>' + '<IMG_CONTEXT>' * (tiles * image_tokens) + '</img>'
    return text.replace('<image>', replacement, 1)

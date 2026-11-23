"""Normalize shared protocol messages for the official multimodal template."""
def native_messages(history):
    messages = []
    for message in history:
        role, content = message['role'], message['content']
        if role not in ('system', 'user', 'assistant'):
            raise ValueError(f'Unsupported chat role: {role}')
        if isinstance(content, str):
            content = [{'type': 'text', 'text': content}]
        else:
            content = [dict(part) for part in content]
        if not content or any(part['type'] not in ('text', 'image') for part in content):
            raise ValueError('Expected text/image content')
        if role != 'user' and any(part['type'] != 'text' for part in content):
            raise ValueError('Only user messages can contain an image')
        messages.append({'role': role, 'content': content})
    roles = [m['role'] for m in messages]
    turns = roles[1:] if roles[:1] == ['system'] else roles
    if turns not in (['user'], ['user', 'assistant', 'user']):
        raise ValueError('Use an original question, optionally followed by its real answer and feedback')
    if sum(p['type'] == 'image' for m in messages for p in m['content']) != 1:
        raise ValueError('Exactly one original image is required')
    return messages

"""InternVL dynamic tiles and ImageNet normalization, without altering raw images."""
from PIL import Image


def image_tiles(image, *, image_size=448, min_tiles=1, max_tiles=12, thumbnail=True):
    if not 1 <= min_tiles <= max_tiles or image_size < 1:
        raise ValueError('Invalid InternVL image tiling settings')
    width, height = image.size
    candidates = {(cols, rows) for n in range(min_tiles, max_tiles + 1)
                  for cols in range(1, n + 1) for rows in range(1, n + 1)
                  if min_tiles <= cols * rows <= max_tiles}
    best, distance = (1, 1), float('inf')
    for cols, rows in sorted(candidates, key=lambda pair: pair[0] * pair[1]):
        error = abs(width / height - cols / rows)
        if error < distance or (error == distance and width * height > .5 * image_size**2 * cols * rows):
            best, distance = (cols, rows), error
    cols, rows = best
    resized = image.resize((cols * image_size, rows * image_size))
    tiles = [resized.crop((col * image_size, row * image_size,
                           (col + 1) * image_size, (row + 1) * image_size))
             for row in range(rows) for col in range(cols)]
    resized.close()
    if thumbnail and len(tiles) > 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


def pixels(image, config):
    import torch
    from torchvision import transforms as T
    size = config['image_size']
    transform = T.Compose([T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC),
                           T.ToTensor(), T.Normalize((.485, .456, .406), (.229, .224, .225))])
    rgb = image.convert('RGB')
    tiles = image_tiles(rgb, image_size=size, min_tiles=config['min_dynamic_patch'],
                        max_tiles=config['max_dynamic_patch'], thumbnail=config['use_thumbnail'])
    try:
        return torch.stack([transform(tile) for tile in tiles])
    finally:
        rgb.close()
        for tile in tiles:
            tile.close()

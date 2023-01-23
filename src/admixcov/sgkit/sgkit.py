import sgkit

def create_tile_masks(ds, type: str, size: int=1000):
    if type == 'variant':
        wds = sgkit.window_by_variant(ds, size=size)
    elif type == 'position':
        wds = sgkit.window_by_position(ds, size=size)
    variant_index = [i for i, x in enumerate(wds.variant_id.values)]
    tile_masks = [
        (start <= variant_index) & (variant_index < stop)
		for start, stop in zip(wds.window_start.values, wds.window_stop.values)
    ]
    return tile_masks
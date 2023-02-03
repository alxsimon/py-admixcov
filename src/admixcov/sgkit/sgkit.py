import sgkit
import numpy as np

def create_tile_idxs(ds, type: str, size: int=1000):
    if type == 'variant':
        wds = sgkit.window_by_variant(ds, size=size)
    elif type == 'position':
        wds = sgkit.window_by_position(ds, size=size)
    variant_index = [i for i, _ in enumerate(wds.variant_id.values)]
    tile_masks = [
        np.where(
            (start <= variant_index) & (variant_index < stop)
        )[0]
		for start, stop in zip(wds.window_start.values, wds.window_stop.values)
        if start != stop # filter out empty tiles
    ]
    return tile_masks
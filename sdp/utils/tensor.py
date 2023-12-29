from typing import Optional

import torch


def stack_imgs_maps(maps: dict[str, list[torch.Tensor]], maps_names: list[str], imgs: Optional[list[torch.Tensor]] = None) -> torch.Tensor:
    rows = []
    n = len(maps[maps_names[0]])
    for i in range(n):
        row = []
        if imgs is not None:
            row.append(imgs[i])
        for map_name in maps_names:
            row.append(maps[map_name][i])
        row = torch.concatenate(row, dim=2)
        rows.append(row.permute(2, 0, 1))

    rows = torch.stack(rows, dim=0)
    return rows



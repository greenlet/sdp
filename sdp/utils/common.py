import re
from typing import Union, Optional

SplitsType = Union[int, list[int], tuple[int, ...], float, list[float], tuple[float, ...]]
IntPat = re.compile(r'^\s*(\d+)\s*$')

def is_iter_len(x: object) -> bool:
    return hasattr(x, '__iter__') and hasattr(x, '__len__')


def split_range(n: int, splits: SplitsType) -> list[int]:
    isitlen = is_iter_len(splits)
    isfloat = False
    if isitlen:
        isfloat = any(type(x) == float for x in splits)
        splits = [float(x) for x in splits]

    res = [0]
    if type(splits) == int:
        assert splits == -1 or 0 < splits <= n
        if splits == -1:
            return [0, n]
        div, rem = divmod(n, splits)
        i_split, i = 0, 0
        while i_split < splits:
            off = div
            if rem > 0:
                off += 1
                rem -= 1
            i += off
            res.append(i)
            i_split += 1
        return res

    if isitlen and len(splits) == 0:
        return [0, n]

    if type(splits) == float:
        splits = [splits]

    if isitlen and type(splits[0]) == float:
        spl = []
        was_neg, total = False, 0
        for s in splits:
            if s < 0:
                spl.append(-1)
                was_neg = True
            else:
                x = int(n * s)
                spl.append(x)
                total += x
        if not was_neg and total < n:
            spl.append(n - total)
        splits = spl

    if isitlen and type(splits[0]) == int:
        was_neg, total = False, 0
        for s in splits:
            assert type(s) == int
            if s == -1:
                assert not was_neg
                was_neg = True
            else:
                assert s > 0
                total += s
        assert was_neg and total < n or total == n, f'was_neg: {was_neg}. total: {total}. n: {n}'
        i, rest = 0, n - total
        for s in splits:
            i += (s if s > 0 else rest)
            res.append(i)
        return res

    raise Exception(f'Unknown splits format: {splits}')


def parse_int_opt(s: str) -> Optional[int]:
    m = IntPat.match(s)
    if not m:
        return
    return int(m.group(1))


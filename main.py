import numpy as np


with open('requirements.txt', 'r') as f:
    lines = f.readlines()

lines = [l.strip() for l in lines]
lines = list(filter(lambda x: x, lines))
lines_lw = [l.lower() for l in lines]
inds = np.argsort(lines_lw)
lines = [lines[i] for i in inds]
print(inds)
print(lines)

with open('requirements.txt', 'w') as f:
    f.write('\n')
    f.write('\n'.join(lines))
    f.write('\n' * 2)


# def fn():
#     try:
#         print('try!')
#         y = None
#         x = 2 / (y - 3)
#     finally:
#         print('finally!')

# print('fn():')
# fn()

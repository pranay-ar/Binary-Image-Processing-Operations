from scipy.spatial import ConvexHull
from skimage import data,img_as_float
from skimage.util import invert
import numpy as np
from skimage.measure.pnpoly import grid_points_in_poly
from PIL import Image as im
from PIL import  ImageEnhance

image=invert(data.horse())
n =np.ascontiguousarray(image, dtype=np.uint8)
rows,cols=n.shape
ndim = image.ndim
coords = np.ones((2 * (rows + cols), 2), dtype=np.intp)
coords *= -1
nonzero = coords
rows_cols = rows + cols
rows_2_cols = 2 * rows + cols

for r in range(rows):
    rows_cols_r = rows_cols + r
    for c in range(cols):
        if n[r, c] != 0:
            rows_c = rows + c
            rows_2_cols_c = rows_2_cols + c
            if nonzero[r, 1] == -1:
                nonzero[r, 0] = r
                nonzero[r, 1] = c
            elif nonzero[rows_cols_r, 1] < c:
                nonzero[rows_cols_r, 0] = r
                nonzero[rows_cols_r, 1] = c
            if nonzero[rows_c, 1] == -1:
                nonzero[rows_c, 0] = r
                nonzero[rows_c, 1] = c
            elif nonzero[rows_2_cols_c, 0] < r:
                nonzero[rows_2_cols_c, 0] = r
                nonzero[rows_2_cols_c, 1] = c

coords =coords[coords[:, 0] != -1]


from itertools import product
offsets = np.zeros((2 * image.ndim, image.ndim))
for vertex, (axis, offset) in enumerate(product(range(image.ndim), (-0.5, 0.5))):
    offsets[vertex, axis] = offset
coords = (coords[:, np.newaxis, :] + offsets).reshape(-1,ndim)


def unique_rows(ar):
    ar = np.ascontiguousarray(ar)
    ar_row_view = ar.view('|S%d' % (ar.itemsize * ar.shape[1]))
    _, unique_row_indices = np.unique(ar_row_view, return_index=True)
    ar_out = ar[unique_row_indices]
    return ar_out
coords = unique_rows(coords)
hull = ConvexHull(coords)
vertices = hull.points[hull.vertices]
mask = grid_points_in_poly(image.shape, vertices)
mask_data = im.fromarray(mask)
mask_data.save('mask.png')
chull_diff = img_as_float(mask.copy())
chull_diff[image] = 2
chull_data = im.fromarray(chull_diff)
chull_data = chull_data.convert('L')
chull_im = ImageEnhance.Brightness(chull_data)
chull_im.enhance(150).save('final.png')


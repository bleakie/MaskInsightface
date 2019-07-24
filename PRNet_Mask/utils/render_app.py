import numpy as np
from PRNet_Mask.utils.render import vis_of_vertices, render_texture
from scipy import ndimage

from .cython import mesh_core_cython

def crender_colors(vertices, triangles, colors, h, w, c=3, BG=None):
    """ render mesh with colors
    Args:
        vertices: [nver, 3]
        triangles: [ntri, 3]
        colors: [nver, 3]
        h: height
        w: width
        c: channel
        BG: background image
    Returns:
        image: [h, w, c]. rendered image./rendering.
    """

    if BG is None:
        image = np.zeros((h, w, c), dtype=np.float32)
    else:
        assert BG.shape[0] == h and BG.shape[1] == w and BG.shape[2] == c
        image = BG
    depth_buffer = np.zeros([h, w], dtype=np.float32, order='C') - 999999.

    # to C order
    vertices = vertices.T.astype(np.float32).copy(order='C')
    triangles = triangles.T.astype(np.int32).copy(order='C')
    colors = colors.T.astype(np.float32).copy(order='C')

    mesh_core_cython.render_colors_core(
        image, vertices, triangles,
        colors,
        depth_buffer,
        vertices.shape[0], triangles.shape[0],
        h, w, c
    )
    return image

def get_visibility(vertices, triangles, h, w):
    triangles = triangles.T
    vertices_vis = vis_of_vertices(vertices.T, triangles, h, w)
    vertices_vis = vertices_vis.astype(bool)
    for k in range(2):
        tri_vis = vertices_vis[triangles[0, :]] | vertices_vis[triangles[1, :]] | vertices_vis[triangles[2, :]]
        ind = triangles[:, tri_vis]
        vertices_vis[ind] = True
    # for k in range(2):
    #     tri_vis = vertices_vis[triangles[0,:]] & vertices_vis[triangles[1,:]] & vertices_vis[triangles[2,:]]
    #     ind = triangles[:, tri_vis]
    #     vertices_vis[ind] = True
    vertices_vis = vertices_vis.astype(np.float32)  # 1 for visible and 0 for non-visible
    return vertices_vis


def get_uv_mask(vertices_vis, triangles, uv_coords, h, w, resolution):
    triangles = triangles.T
    vertices_vis = vertices_vis.astype(np.float32)
    uv_mask = render_texture(uv_coords.T, vertices_vis[np.newaxis, :], triangles, resolution, resolution, 1)
    uv_mask = np.squeeze(uv_mask > 0)
    uv_mask = ndimage.binary_closing(uv_mask)
    uv_mask = ndimage.binary_erosion(uv_mask, structure=np.ones((4, 4)))
    uv_mask = ndimage.binary_closing(uv_mask)
    uv_mask = ndimage.binary_erosion(uv_mask, structure=np.ones((4, 4)))
    uv_mask = ndimage.binary_erosion(uv_mask, structure=np.ones((4, 4)))
    uv_mask = ndimage.binary_erosion(uv_mask, structure=np.ones((4, 4)))
    uv_mask = uv_mask.astype(np.float32)

    return np.squeeze(uv_mask)


def get_depth_image(vertices, triangles, h, w, isShow = False):
    z = vertices[:, 2:]
    if isShow:
        z = z/max(z)
    depth_image = crender_colors(vertices.T, triangles.T, z.T, h, w, 1)
    # depth_image = render_texture(vertices.T, z.T, triangles.T, h, w, 1) # time is so large
    depth_image = np.squeeze(depth_image)
    depth_image = depth_image / 255.
    return np.squeeze(depth_image)

def faceCrop(img, maxbbox, scale_ratio=2):
    '''
    crop face from image, the scale_ratio used to control margin size around face.
    using a margin, when aligning faces you will not lose information of face
    '''
    xmin, ymin, xmax, ymax = maxbbox
    hmax, wmax, _ = img.shape
    x = (xmin + xmax) / 2
    y = (ymin + ymax) / 2
    w = (xmax - xmin) * scale_ratio
    h = (ymax - ymin) * scale_ratio
    # new xmin, ymin, xmax and ymax
    xmin = x - w / 2
    xmax = x + w / 2
    ymin = y - h / 2
    ymax = y + h / 2

    xmin = max(0, int(xmin))
    ymin = max(0, int(ymin))
    xmax = min(wmax, int(xmax))
    ymax = min(hmax, int(ymax))
    face = img[ymin:ymax, xmin:xmax, :]
    return face

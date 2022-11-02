import numpy as np
from cellworld import *
from scipy.spatial.distance import cdist
from tqdm.notebook import tqdm
import rtree
import matplotlib.path as mpath
import matplotlib.patches as mpatches

def get_vis(e):
    """Gets world visibility from experiment object."""
    w = World.get_from_parameters_names('hexagonal','canonical',e.occlusions)
    occlusion_locations = w.cells.occluded_cells().get("location")
    occlusions_polygons = Polygon_list.get_polygons(occlusion_locations, w.configuration.cell_shape.sides, w.implementation.cell_transformation.size / 2, w.implementation.space.transformation.rotation + w.implementation.cell_transformation.rotation) # ploygon object
    vis = Location_visibility(occlusions_polygons)
    return vis,w

def get_vertices(e):
    """Gets unique vertices from all polygons."""
    # make a list of all polygon vertices
    w = World.get_from_parameters_names('hexagonal','canonical',e.occlusions)
    all_polygons = Polygon_list.get_polygons(w.cells.get('location'),w.configuration.cell_shape.sides, w.implementation.cell_transformation.size / 2, w.implementation.space.transformation.rotation + w.implementation.cell_transformation.rotation)
    x = []
    y = []
    for poly in all_polygons:
        x.append(poly.vertices.get('x'))
        y.append(poly.vertices.get('y'))
    x = np.hstack(x).reshape(1,-1).T
    y = np.hstack(y).reshape(1,-1).T
    verts = np.concatenate((x,y),axis=1)
    pts = verts.tolist()

    # get unique vertices, removing those closeby
    sparse_pts = sparse_subset(pts,0.01)
    sparse_arr = np.vstack(sparse_pts)
    return sparse_arr

def get_tiles(n,e):
    """Get nxn locations tiled across the world in experiment object, then removes tiles
    that are within obstacles in the world. (needs to Display the world to do so)"""
    # generate world tiles
    w = World.get_from_parameters_names('hexagonal','canonical',e.occlusions)
    x = np.linspace(0,1,n)
    xv,yv = np.meshgrid(x,x,indexing='ij')
    xv = xv.reshape(1,-1)
    yv = yv.reshape(1,-1)
    points = np.concatenate((xv,yv)).T

    # get the wall limits
    d = Display(w, fig_size=(1,1), padding=0, cell_edge_color="lightgrey")
    path = d.habitat_polygon.get_path()
    transform = d.habitat_polygon.get_patch_transform()
    newpath = transform.transform_path(path)
    polygon = mpatches.PathPatch(newpath)
    inside = []
    inside.append(~newpath.contains_points(points))

    # get the occlusion limits and remove points
    for poly in d.cell_polygons:
        if poly._facecolor[0]==0:
            path = poly.get_path()
            transform = poly.get_patch_transform()
            newpath = transform.transform_path(path)
            polygon = mpatches.PathPatch(newpath)
            inside.append(newpath.contains_points(points,radius=0.025))
    index = np.any(np.vstack(inside).T,axis=1)
    return points[~index,:]

def plot_tiles(pts,sparse_arr,e):
    a = 1
    w = World.get_from_parameters_names('hexagonal','canonical',e.occlusions)

    # display
    fig,ax = plt.subplots(1,2,figsize=(10,5))
    d = Display(w, fig_size=(5,5), padding=0, cell_edge_color="lightgrey",ax=ax[0])
    ax[0].scatter(pts[:,0],pts[:,1],5,'g',alpha = a)
    ax[0].scatter(sparse_arr[:,0],sparse_arr[:,1],20,'m')

    Display(w, fig_size=(5,5), padding=0, cell_edge_color="lightgrey",ax=ax[1])
    ax[1].scatter(pts[:,0],pts[:,1],5,'g',alpha = a)
    ax[1].scatter(sparse_arr[:,0],sparse_arr[:,1],20,'m')
    ax[1].set_xlim((.25,.3))
    ax[1].set_ylim((.25,.3))
    return [fig,ax]

def dist(p,q):
    """Return distance between two points."""
    return math.hypot(p[0]-q[0],p[1]-q[1])

def sparse_subset(points,r):
    """Return a maximal list of elements of points such that no pairs of
    points in the result have distance less than r."""
    result = []
    index = rtree.index.Index()
    for i, p in enumerate(points):
        px, py = p
        nearby = index.intersection((px - r, py - r, px + r, py + r))
        if all(dist(p, points[j]) >= r for j in nearby):
            result.append(p)
            index.insert(i, (px, py, px, py))
    return result

def compute_visibility(src,dst,vis):
    """returns a visibility graph (V) and angle graph (A) between every point in
    src and dst according to visibility object vis"""
    m = len(src)
    p = len(dst)
    V = np.empty((m,p))
    A = np.empty((m,p))
    for i in tqdm(range(m)):
        loci = Location(src[i,0],src[i,1])
        for j in range(p):
            locj = Location(dst[j,0],dst[j,1])
            V[i,j] = vis.is_visible(loci,locj)
            A[i,j] = to_degrees(loci.atan(locj))
    return V,A

import numpy as np
from cellworld import *
from scipy.spatial.distance import cdist
from tqdm.notebook import tqdm
import rtree
import matplotlib.path as mpath
import matplotlib.patches as mpatches


def get_closest_tile(loc,vis_graph):
    """Get closest tile location to given location."""
    idx = cdist(loc.reshape(1,-1),vis_graph['src']).argmin()
    return vis_graph['src'][idx,:],idx


def get_fov_points(head_pt,head_angle,vis_graph,fov = np.rad2deg(1.74533)):
    """Get tiles within the FOV."""
    _,idx = get_closest_tile(head_pt,vis_graph)
    tA = np.deg2rad(vis_graph['A'][idx,:])
    sA = np.deg2rad(head_angle)
    a = np.min(np.concatenate(((2 * np.pi) - np.abs(tA - sA).reshape(-1,1),
                               np.abs(tA - sA).reshape(-1,1)),axis=1),axis=1)
    return (np.abs(np.rad2deg(a)) < fov)


def get_visibility(loc,vis_graph):
    """Returns which points are visible from the provided location"""
    pt,idx = get_closest_tile(loc,vis_graph)
    return (vis_graph['V'][idx,:]==1),pt


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


def compute_itor(pose,head_angle,vis_graph,head_parts=['head_base'],body_parts=['body_mid'],fov = np.rad2deg(1.74533)):
    """Returns ITOR value and various visible points in the environment based on
    pose, head angle, FOV"""
    # get closest tiles for each part:
    v = []
    ppoints = []
    for p in pose:
        vis,pt = get_visibility(np.array(pose[p]),vis_graph)
        v.append(vis)
        ppoints.append(pt)
    v = np.vstack(v)
    ppoints = np.vstack(ppoints)

    # get head and body parts
    head_ind = [i for i in range(len(list(pose.keys()))) if list(pose.keys())[i] in head_parts]
    body_ind = [i for i in range(len(list(pose.keys()))) if list(pose.keys())[i] in body_parts]

    # get points in the FOV
    fov_verts = get_fov_points(ppoints[head_ind,:],head_angle,vis_graph,fov)

    # which vertices that are visible to the head:
    vis_verts = np.any(v[head_ind,:],axis=0) & fov_verts

    # of FOV visible vertices, which ones are visible to the body:
    body_verts = np.any(v[body_ind,:],axis=0) & vis_verts

    # compute ITOR as 1-(visible_to_body/visible_to_head)
    ITOR = 1 - np.sum(body_verts)/np.sum(vis_verts)

    return {'ITOR':ITOR,
            'vis_omni': np.any(v[head_ind,:],axis=0),
            'vis_head': vis_verts,
            'vis_body': body_verts,
            'head_parts': head_parts,
            'head_idx': head_ind,
            'body_parts': body_parts,
            'body_idx': body_ind,
            'pose_points': ppoints,
            'head_angle': head_angle}

def compute_itor_pose(pose,head_angle,vis_graph,head_parts=['head_base'],body_parts=['body_mid'],fov = np.rad2deg(1.74533)):
    """Returns ITOR value pose points in the environment based on
    pose, head angle, FOV"""
    # get closest tiles for each part:
    v = []
    ppoints = []
    part = []
    for p in pose:
        if p.score > 0.8:
            part.append(p.part)
            vis,pt = get_visibility(np.array([p.location.x,p.location.y]),vis_graph)
            v.append(vis)
            ppoints.append(pt)

    if v:
        v = np.vstack(v)
        ppoints = np.vstack(ppoints)

        # get head and body parts
        head_ind = [i for i in range(len(part)) if part[i] in head_parts]
        body_ind = [i for i in range(len(part)) if part[i] in body_parts]

        if head_ind:
            # get points in the FOV
            fov_verts = get_fov_points(ppoints[head_ind,:],head_angle,vis_graph,fov)

            # which vertices that are visible to the head:
            vis_verts = np.any(v[head_ind,:],axis=0) & fov_verts

            # of FOV visible vertices, which ones are visible to the body:
            body_verts = np.any(v[body_ind,:],axis=0) & vis_verts

            # compute ITOR as 1-(visible_to_body/visible_to_head)
            ITOR = 1 - np.sum(body_verts)/np.sum(vis_verts)

        else:
            ITOR = np.nan
            ppoints = np.array([np.nan,np.nan])
    else:
        ITOR = np.nan
        ppoints = np.array([np.nan,np.nan])

    return {'ITOR':ITOR,
            'pose_points': ppoints}

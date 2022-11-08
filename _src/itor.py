import numpy as np
from cellworld import *
from scipy.spatial.distance import cdist
from tqdm.notebook import tqdm
import rtree
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import os
from random import choices
import pandas as pd

from .pose import *
from .multiproc import *


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

            # get vertices that are visible to the head:
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


def compute_itor_null(
    log,
    poselib,
    vis_graph,
    d,
    outpath = './_data/results',
    k = 500,
    score_cutoff = 0.8,
    dist_cutoff = 0.00,
    body_parts = ['body_mid','tail_base','tail_post_base','tail_pre_tip','tail_tip'],
    start_ep = 0):

    '''
    Computes a null distribution of ITOR values
    Inputs:
    - log: an experiment log file containing pose information
    - poselib: a poselibrary containing many different poses
    - vis_graph: a visibility graph for the experiment world
    - d: a display object for the world
    - outpath: where to save the .pkl results
    - k: number of null samples (500 default)
    - score_cutoff: camera score to exclude frames with poor tracking (0.8 default)
    - dist_cutoff: distance from start to exclude frames (0.00 default)
    - body_parts: the body parts to include in the ITOR computation (all by default)
    - start_ep: the episode to start looping over (default is 0, use for debugging only)
    '''

    # setup
    l = log
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    print(l)
    # get the poses for this experiment
    p = poselib.loc[poselib.log_file == l]
    if len(p) > 0:

        # get the pose library for this mouse
        pl = poselib.loc[poselib.mouse == p.iloc[0].mouse]

        # for each episode
        uep = p.episode.unique()
        for ep in range(start_ep, uep.shape[0]):
        #for ep in tqdm(range(uep.shape[0])):
            print(f' Episode {ep}')

            # check if the file exists
            fn = l.split('\\')[-1].replace('experiment.json',f'episode{ep:03d}.pkl',1)
            filepath = os.path.join(outpath,fn)
            if not os.path.exists(filepath):

                # get frames from this episode
                episode = p.loc[(p.episode == uep[ep])]

                # define poses to sample from for this mouse
                I = ((pl.episode != uep[ep]) | \
                     (pl.session != episode.session.unique()[0])) & \
                     (pl.score_mean > score_cutoff) & \
                     (pl.pose_ordered)

                # loop through frames
                D = {} # dictionary to store results
                dcnt = 0
                for index,row in tqdm(episode.iterrows(), total=episode.shape[0]):
                #for index,row in episode.iterrows():

                    # if the true pose is in the arena and has good tracking
                    if pose_inside_arena(row.pose,d) & \
                    (row.start_dist > dist_cutoff) & \
                    (row.score_mean > score_cutoff) & \
                    (row.pose_ordered):

                        #print(f'  Frame {row.frame}/{episode.frame.max()}',end=' ')
                        #pbar = tqdm(total=k)

                        # compute true ITOR
                        itor = compute_itor_pose(row.pose,
                          row.head_angle,
                          vis_graph,
                          head_parts=['head_base'],
                          body_parts=body_parts)

                        # generate a null ITOR distribution
                        cnt = 0
                        null_pose_index = []
                        null_pose_itor = []
                        null_pose = []
                        while (cnt < k):

                            # get a pose sample
                            samp = choices(np.argwhere(np.array(I)),k=1)[0]
                            pose_samp = pl.iloc[samp].pose.item()

                            # transform it
                            pose_null,src_angle,src_loc,ref_angle,ref_loc = \
                            match_pose(row.pose,pose_samp)

                            # check if pose sample is in the arena, not in obstacles, and ordered
                            # good_sample = True
                            good_sample = pose_inside_arena(pose_null,d) and \
                                          not pose_inside_occlusions(pose_null,d)

                            # if a good sample, compute ITOR
                            if good_sample:
                                itor_null = compute_itor_pose(pose_null,
                                  row.head_angle,
                                  vis_graph,
                                  head_parts=['head_base'],
                                  body_parts=body_parts)
                                null_pose_index.append(pl.iloc[samp].index)
                                null_pose_itor.append(itor_null['ITOR'])
                                null_pose.append(itor_null['pose_points'])
                                #pbar.update(1)
                                cnt = cnt + 1

                        # add to dictionary
                        row['ITOR'] = itor['ITOR']
                        row['null_ITOR'] = null_pose_itor
                        row['null_index'] = null_pose_index
                        row['null_pose'] = null_pose
                        D[dcnt] = row.to_dict()
                        dcnt = dcnt + 1
                        #pbar.close

                # convert the episode dictionary to a dataframe and save
                print('  saving...')
                df = pd.DataFrame.from_dict(D,'index')
                df.to_pickle(filepath)

            else:
                print(f'{filepath} exists... skipping...')


def compute_itor_null_fast(args):
    log,shared_poselib,shared_A,shared_V,shared_src,shared_dst = args

    # read everything from shared memory
    poselib = shared_poselib.read()
    A = shared_A.read()
    V = shared_V.read()
    src = shared_src.read()
    dst = shared_dst.read()

    # build dictionary as expected by compute_itor_null
    vis_graph = {'V':V,'A':A,'src':pts,'dst':sparse_arr}

    # get the display
    e = Experiment.load_from_file(log)
    vis,w = get_vis(e)
    d = Display(w)

    print(log)

    # call the old function
    compute_itor_null(log,poselib,vis_graph,d,k=1,outpath='./_data/mptest')

    return True

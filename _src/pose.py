from cellworld import *
import numpy as np
import glob
from tqdm.notebook import tqdm
import pandas as pd
import matplotlib.path as mpath
import matplotlib.patches as mpatches

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def build_pose_library(logs,filename='./_data/logs/_pose_library.pkl'):
    '''
    Builds a pose library from a list of .json logs
    Inputs:
    - logs: list of .json logs
    - filename: .pkl file name for dataframe
    Returns:
    - df: dataframe where each row represents a frame where the mouse was visible
        'mouse': mouse ID
        'session': date and time of experiment
        'experiment_type': experiment type
        'world': world ID
        'episode': episode number
        'frame': frame number
        'pose': pose object (JsonList)
        'pose_norm': normalized pose object (head facing east, centered at (0,0))
        'head_angle': head angle of the mouse
        'pose_ordered': whether this pose is in reasonable order (body behind nose, roughly)
        'start_dist': distance from start of habitat (Location(0.01,0.5))
        'score_mean': mean of pose scores
        'score_std': std of pose scores
        'score_max': max of pose scores,
        'predator_location': Location of predator
        'predator_angle': orientation of predator
        'log_file': location of the log file for this frame}
    '''

    if glob.glob(filename):
        print(f'{filename} found, loading...')
        df = pd.read_pickle(filename)
        return df

    start_location = Location(0.01,0.5)

    d = {}
    cnt = 0
    for i in tqdm(range(len(logs))):
        # load experiment
        e = Experiment.load_from_file(logs[i])

        # get filename
        fn = e.name.split('_')
        mouse = fn[3]
        session = '_'.join(fn[1:3])
        world = '_'.join(fn[4:6])
        experiment_type = fn[-1]

        # print(e.name)
        # print('\t',end='')

        for j,ep in enumerate(e.episodes):
            # get episode and trajectories
            episode = j
            pt = ep.trajectories.where('agent_name','prey').get_unique_steps()
            rt = ep.trajectories.where('agent_name','predator')
            # print(j,end=' ')

            # for each frame
            for step in pt:
                if step.data:
                    # prey info
                    frame = step.frame
                    pose = PoseList.parse(step.data)
                    head_angle = step.rotation
                    start_dist = pose[1].location.dist(start_location)
                    score_mean = np.array([p.score for p in pose]).mean()
                    score_std = np.array([p.score for p in pose]).std()
                    score_max = np.array([p.score for p in pose]).max()
                    pose_ordered = pose_is_ordered(pose)

                    # predator info
                    rind = np.where(np.array(rt.get('frame'))==step.frame)[0]
                    if len(rind) > 0:
                        predator_location = rt[rind[0]].location
                        predator_angle = rt[rind[0]].rotation
                    else:
                        predator_location = np.nan
                        predator_angle = np.nan

                    # append to dict
                    d[cnt] = {
                        'mouse': mouse,
                        'session': session,
                        'experiment_type': experiment_type,
                        'world': world,
                        'episode': episode,
                        'frame': frame,
                        'pose': pose,
                        'pose_norm': normalize_pose(pose),
                        'head_angle': head_angle,
                        'pose_ordered': pose_ordered,
                        'start_dist': start_dist,
                        'score_mean': score_mean,
                        'score_std': score_std,
                        'score_max': score_max,
                        'predator_location': predator_location,
                        'predator_angle': predator_angle,
                        'log_file': logs[i]}
                    cnt = cnt + 1
        # print('\n')

    # save to file
    print(f'saving pose library to {filename}')
    df = pd.DataFrame.from_dict(d,'index')
    df.to_pickle(filename)

    return df

# pose functions
def get_pose_dict(step):
    '''Convert step.data to dict'''
    p = PoseList.parse(step.data)
    pose = {}
    for i in range(len(p)):
        pose.update({p[i].part:[p[i].location.x,p[i].location.y]})
    return pose

def pose2array(pose,score=0.8):
    '''Covert pose to array'''
    return np.vstack([[i.location.x,i.location.y] for i in pose if i.score > score])

def get_pose_angle(pose, parts=['head_base','nose']):
    '''Get angle between two parts of pose object'''
    posec = pose.copy()
    loca = [i.location for i in posec if i.part==parts[0]][0]
    locb = [i.location for i in posec if i.part==parts[1]][0]
    #loca.y = loca.y*-1
    #locb.y = locb.y*-1
    angle = to_degrees(loca.atan(locb))
    return angle

def transform_pose(pose, origin=Location(0,0), offset=Location(0,0), angle=0):
    '''Transforms a pose object by offsetting and rotating around an origin'''
    pose_norm = pose.copy()
    for i,p in enumerate(pose_norm):
        p.location = p.location + offset
        r = rotate([p.location.x,p.location.y],
                   origin=[origin.x,origin.y],
                   degrees=-angle)
        p.location = Location(r[0],r[1])
    return pose_norm

def match_pose(pose0,pose1,ref_part='head_base'):
    '''Transforms pose1 to match pose0 based on reference part location and head angle'''
    # get reference location and head angle
    ref_loc = [i.location for i in pose0 if i.part==ref_part][0]
    ref_angle = get_pose_angle(pose0)

    # get the source location and head angle
    src_loc = [i.location for i in pose1 if i.part==ref_part][0]
    src_angle = get_pose_angle(pose1)

    # calculate location and angle offset
    a = ref_angle - src_angle
    if a > 180:
        a -= 360
    if a < -180:
        a += 360
    offset = (ref_loc - src_loc)

    # offset and rotate
    pose_norm = transform_pose(pose1,origin=ref_loc,offset=offset,angle=a)

    return pose_norm,src_angle,src_loc,ref_angle,ref_loc

def plot_pose(pose,ax=plt,**plt_kwargs):
    '''Plots pose'''
    ppoints = []
    npoint = []
    hpoint = []
    for p in pose:
        if 'nose' in p.part:
            npoint = [p.location.x,p.location.y]
        elif 'head' in p.part:
            hpoint = [p.location.x,p.location.y]
        else:
            ppoints.append([p.location.x,p.location.y])
    ppoints = np.vstack(ppoints)
    hpoint = np.hstack(hpoint)
    npoint = np.hstack(npoint)
    h = []
    h.append(ax.scatter(ppoints[:,0],ppoints[:,1],**plt_kwargs))
    h.append(ax.plot(hpoint[0],hpoint[1],'*',color=h[0].get_facecolors()[0]))
    h.append(ax.plot(npoint[0],npoint[1],'^',color=h[0].get_facecolors()[0]))
    return(h)

def rotate(p, origin=(0, 0), degrees=0):
    '''Rotates point p around [origin] by [degrees]'''
    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T-o.T) + o.T).T)

def pose_inside_arena(pose, d, cutoff = 0):
    '''Returns true if all parts are inside the arena'''
    pose_points = pose2array(pose,score=cutoff)
    path = d.habitat_polygon.get_path()
    transform = d.habitat_polygon.get_patch_transform()
    newpath = transform.transform_path(path)
    polygon = mpatches.PathPatch(newpath)
    inside = []
    inside.append(newpath.contains_points(pose_points))
    return np.all(np.vstack(inside))

def pose_inside_occlusions(pose, d, cutoff = 0):
    '''Returns true if any part is in an occlusion'''
    pose_points = pose2array(pose,score=cutoff)
    inside = []
    for poly in d.cell_polygons:
        if poly._facecolor[0]==0:
            path = poly.get_path()
            transform = poly.get_patch_transform()
            newpath = transform.transform_path(path)
            polygon = mpatches.PathPatch(newpath)
            inside.append(newpath.contains_points(pose_points))
    return np.any(np.vstack(inside))

def pose_is_ordered(pose):
    '''Checks if pose parts are in roughly the right order'''
    posen = normalize_pose(pose)
    return all([i.location.x <= 0 for i in posen if not 'nose' in i.part])

def normalize_pose(pose):
    '''Rotates mouse to face left->right with head at origin'''
    ref_loc = [i.location for i in pose if i.part=='head_base'][0]
    ref_angle = 90 - get_pose_angle(pose)
    offset = Location(0,0) - ref_loc
    return transform_pose(pose,origin=Location(0,0),offset=offset,angle=ref_angle)


# json classes
class PoseList(JsonList):
  def __init__(self):
    super().__init__(list_type=PosePart)

class PosePart(JsonObject):
  def __init__(self):
    self.part = str()
    self.location = Location()
    self.camera = int()
    self.score = float()

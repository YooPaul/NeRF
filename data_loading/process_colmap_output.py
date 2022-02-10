import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
import imageio
import skimage
import os


from read_write_model import read_model

def parse_colmap_output(path, extention):
    cams, imgs, points3D = read_model(path=path, ext=extention)

    cameras = {}
    images = {}

    for i in cams:
        cam = cams[i]
        camera_id = cam.id
        intrinsics = {}

        intrinsics['W'] = cam.width
        intrinsics['H'] = cam.height

        intrinsics['f'] = cam.params[0]
        intrinsics['cx'] = cam.params[1]
        intrinsics['cy'] = cam.params[2]

        intrinsics['k'] = cam.params[3]

        cameras[camera_id] = intrinsics


    for i in imgs:
        img = imgs[i]
        image_name = img.name
        extrinsics = {}

        R = Rotation.from_quat(img.qvec).as_matrix()
        t = img.tvec

        extrinsics['R'] = torch.from_numpy(R) # 3x3
        extrinsics['t'] = torch.from_numpy(t) # 3
        extrinsics['c_id'] = img.camera_id
        
        images[image_name] = extrinsics

    return cameras, images


def get_rays(colmap_path, extention, images_path, ndc=True):

    cameras, images = parse_colmap_output(colmap_path, extention)

    camera_poses = None
    rays = []
    for i, image in enumerate(images):
        print('Image:', i)
        fullpath = os.path.join(images_path, image)
        img = imageio.imread(fullpath) # H x W x C
        img = skimage.img_as_float32(img) # shift range to [0, 1]
        img = torch.from_numpy(img)

        H, W = img.shape[:2]
        #img = transforms.Resize((H // 4, W // 4))(img)
        #H, W = img.shape[:2]

        extrinsics = images[image]
        intrinsics = cameras[extrinsics['c_id']]

        R = extrinsics['R']
        t = extrinsics['t']
        camera_origin = - R.T @ t # size (3)

        # we follow OpenGL coordinate convention
        # where x-axis in camera coordinate space points right
        # y-axis points up and z-axis points backward

        # since the rotation matrix output from COLMAP follows OpenCV convention
        # where x-axis in camera coordinate space points right
        # y-axis points down and z-axis points forward
        # we need to invert up and forward
        if camera_poses is None:
            # convert to cam to world
            camera_poses =  torch.concat((R.T, (-R.T @ t).reshape(3,1)), dim=-1).unsqueeze(0) # (3, 4)
            r = camera_poses[:,0:1]
            u = camera_poses[:,1:2]
            f = camera_poses[:,2:3]
            t = camera_poses[:,3:4]
            camera_poses = torch.concat( (r, -u, -f, t), dim=-1)
        else:
            new_pose = torch.concat((R.T, (-R.T @ t).reshape(3,1)), dim=-1).unsqueeze(0)
            r = new_pose[:,0:1]
            u = new_pose[:,1:2]
            f = new_pose[:,2:3]
            t = new_pose[:,3:4]
            new_pose = torch.concat( (r, -u, -f, t), dim=-1)
            camera_poses = torch.concat((camera_poses, new_pose), dim=0)

        original_H = intrinsics['H']
        f = H / original_H * intrinsics['f']
        cx = W // 2 # intrinsics['cx']
        cy = H // 2 # intrinsics['cy']

        # Radial distortion coefficient
        k = intrinsics['k']

        # From each image sample 70000 rays
        grid_x, grid_y = torch.meshgrid(torch.arange(0, W), torch.arange(0, H), indexing='xy')
        uv = torch.cat(tuple(torch.dstack([grid_x, grid_y]))) # (H*W, 2)

        # Apply inverse intrinsic matrix
        xpp = (uv[:,0] - cx) / f # (H*W)
        ypp = (uv[:,1] - cy) / f

        # Radial distortion correction
        #roots = np.roots([1, 2*k, 1, -(xpp**2 + ypp**2)])
        #r_sq = roots.max().astype(float).item(0)

        #assert(r_sq >= 0)

        # Pixel (u,v) in 3D camera coordinate space
        xp = xpp.reshape(-1,1) #/ (1 + k * r_sq)
        yp = ypp.reshape(-1,1) #/ (1 + k * r_sq)
        zp = torch.ones_like(xp)

        xyz = torch.concat([xp, yp, zp], dim=-1) # (H*W, 3)

        # Pixel (u,v) in 3D world coordinate space
        xyz = torch.bmm(R.T.unsqueeze(0).expand(xyz.shape[0], 3, 3) , (xyz - t).unsqueeze(-1)) # (H*W,3,1)

        # Ray directions
        ray_dirs = xyz.reshape(-1, 3) - camera_origin #  (H*W, 3)
        ray_dirs = ray_dirs / torch.linalg.norm(ray_dirs, dim=1, keepdim=True) # Normalize

        colors = img[uv[:,1], uv[:,0]] # (H*W, 3)
        rays.append((camera_origin, ray_dirs, colors, (f, cx, cy)))

    if ndc:
        # Compute average pose and convert all ray origins and directions to NDC space w.r.t. average camera
        average_c2w = get_avg_camera_pose(camera_poses) # (1, 3, 4)
        for i, (camera_origin, ray_dirs, colors, K) in enumerate(rays):
            ndc_origin, ndc_ray_dir = to_ndc_space(camera_origin, ray_dirs, average_c2w, K)
            rays[i] = (ndc_origin, ndc_ray_dir, colors)

    return rays, H, W


def get_avg_camera_pose(camera_poses):
    '''
    Args:
        camera_poses: A tensor of size (N, 3, 4) containing all camera poses(camera to world) of images of a single object

    Returns:
        Average camera pose 
    '''

    # In the conventional camera coordinate system used by OpenGL, 
    # the forward axis is pointing backwards and up axis points above the camera.
    # Thus, the up vector is a cross between forward and right. 
    average_forward_axis = F.normalize(torch.mean(camera_poses[:, :, 2], dim=0, keepdim=True)) #.expand((camera_poses.shape[0], 3)) # (N, 3)
    
    average_right_axis = F.normalize(torch.mean(camera_poses[:, :, 0], dim=0, keepdim=True)) #.expand((camera_poses.shape[0], 3)) # (N, 3)

    average_up_axis = torch.cross(average_forward_axis, average_right_axis)

    average_camera_pos = torch.mean(camera_poses[:, :, 3], dim=0, keepdim=True) # (1, 3)

    average_camera_orientation = torch.stack([average_right_axis, average_up_axis, average_forward_axis], dim=1) # (1,3,3)

    return torch.concat((average_camera_orientation, average_camera_pos.reshape(1,3,1)), dim=-1) # (1, 3, 4)


def to_ndc_space(camera_origin, ray_directions, average_c2w, K, near=1,far=10):

    f, cx, cy, = K
    R, t = average_c2w[0,:3,:3], average_c2w[0,:3,3]
    # convert to world to camera
    R = R.T # (3,3)
    t = -R.T @ t # (3)
    new_origin = R @ camera_origin + t # (3)
    new_ray_dir = torch.bmm(R.unsqueeze(0).expand(ray_directions.shape[0], 3, 3), ray_directions.reshape(-1, 3, 1)).reshape(-1, 3) # (H*W, 3)

    # shift origin to the near plane
    # now there's a different origin for each ray direction
    tn = -(near + new_origin[2]) / new_ray_dir[:, 2, None] # (HW, 1)
    new_origin = new_origin + tn * new_ray_dir # (H*W, 3)

    # compute origin and ray direction in NDC space
    o_z = new_origin[:, 2].reshape(-1,1)
    mult = torch.ones_like(new_origin)
    mult[...,0] *= -f/cx
    mult[...,1] *= -f/cy 
    mult[...,2] = 1 + 2*near/o_z.flatten()
    ndc_origin = (mult * new_origin) / o_z

    ndc_ray_dir = torch.tensor([-f/cx, -f/cy, 1.0]) * \
                                torch.concat(
                                    (new_ray_dir[:,0, None] / new_ray_dir[:,2, None] - new_origin[:,0, None] / new_origin[:,2, None], 
                                    new_ray_dir[:,1, None] / new_ray_dir[:,2, None] - new_origin[:,1, None] / new_origin[:,2, None],
                                    -2 * near / new_origin[:,2, None]), dim=-1)
    
    return ndc_origin, ndc_ray_dir # (H*W, 3), (H*W, 3)


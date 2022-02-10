import torch

class NeRFDataset(torch.utils.data.Dataset):
    def __init__(self, rays):
        super(NeRFDataset, self).__init__()
        
        self.rays = rays
        self.num_images = len(rays)
        self.rays_per_image = rays[0][0].shape[0]

    def __len__(self):
        # return the number of sequences in the dataset
        return self.num_images * self.rays_per_image
        
    def __getitem__(self, idx):
        #camera_pos, d, color = self.data[idx]
        image_idx = idx // self.rays_per_image
        camera_origins, ray_dirs, colors = self.rays[image_idx]
        ray_idx = idx % self.rays_per_image
        return camera_origins[ray_idx], ray_dirs[ray_idx], colors[ray_idx] 
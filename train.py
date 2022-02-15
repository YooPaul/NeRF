
import torch
import torch.nn.functional as F
import torch.utils.tensorboard as tensorboard
from torchvision import utils

from models import NeRF
from data_loading.process_colmap_output import get_rays
from data_loading.dataset import NeRFDataset
from utils import inverse_transform_sampling, integrate_color

def train(colmap_path, extention, images_path):
    BATCH_SIZE = 300
    EPOCHS = 200
    lr=5e-4

    # Points to sample
    Nc = 64
    Nf = 128
    #tn = 1
    #tf = 10
    # In NDC from 0 to 1
    tn = 0
    tf = 1

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Using device', device)

    coarse_model = NeRF(10, 4, device=device)#.to(device)
    fine_model = NeRF(10, 4, device=device)#.to(device)
    optim = torch.optim.Adam(list(coarse_model.parameters()) + list(fine_model.parameters()), lr=lr)

    # Set up tensorboard writers
    summary_writer = tensorboard.SummaryWriter('/content/logs')#'Tensorboard/NeRF/logs/')

    rays, H, W = get_rays(colmap_path, extention, images_path, ndc=True)
    train_dataset =  NeRFDataset(rays)
    train = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    camera_origin, ray_dirs, colors = rays[0]
    fixed_rays  = (camera_origin.to(device), ray_dirs.to(device), colors.to(device))

    step = 0
    for epoch in range(EPOCHS):
        for idx, (ray_origins, ray_directions, colors) in enumerate(train):
            #torch.cuda.empty_cache()
            optim.zero_grad()
            ray_origins = ray_origins.to(device)
            ray_directions = ray_directions.to(device)
            colors = colors.to(device)


            intervals = torch.linspace(tn, tf, Nc + 1, device=device)
            ts = (intervals[1:] - intervals[:-1]) * torch.rand(BATCH_SIZE, Nc, device=device) + intervals[:-1] # size (Batch_size, Nc)
            ts = ts.reshape(BATCH_SIZE, Nc, 1) # size (Batch_size, Nc, 1)
            points = ray_origins.unsqueeze(1) + ts * ray_directions.unsqueeze(1) # (Batch_size, Nc, 3)
            
            # points -> (Batch_size, Nc, 3)
            # density -> (Batch_size, Nc, 1)
            # color -> (Batch_size, Nc, 3)
            density, color = coarse_model(points, ray_directions)

            # (Batch_size, 3),  (Batch_size, Nc)
            Cc, weights = integrate_color(Nc, density, color, ts, device)

            # Use weights as a PDF(Probability Density Function) to inverse transform sample Nf points
            weights = weights / weights.sum(dim=1, keepdim=True)

            new_ts = inverse_transform_sampling(ts, weights, Nf, device) # (Batch_size, Nf)
            new_ts = new_ts.reshape(BATCH_SIZE, Nf,  1) # (Nf, 1)
            new_points = ray_origins.unsqueeze(1) + new_ts * ray_directions.unsqueeze(1) # (Batch_size, Nf, 3)
            
            # density -> (Batch_size, Nc + Nf, 1)
            # color -> (Batch_size, Nc + Nf, 3)
            density, color = fine_model(torch.concat((points, new_points), dim=1), ray_directions) # concat size (Batch_size, Nc + Nf, 3)
            
            # (Batch_size, 3),  (Batch_size, Nc)
            Cf, weights = integrate_color(Nc + Nf, density, color, torch.sort(torch.concat((ts, new_ts), dim=1),dim=1)[0], device)

            loss = F.mse_loss(Cc, colors) + F.mse_loss(Cf, colors)
            loss.backward()
            optim.step()

            step += 1
            if (step - 1) % 500 == 0:
                print('Epoch: %d/%d\tBatch: %03d/%d\tLoss: %f' % (epoch, EPOCHS, idx, len(train), loss.item()))

            if (step - 1) % 2000 == 0:
                with torch.no_grad():
                    camera_origin, ray_dirs, colors = fixed_rays
                    HW = camera_origin.shape[0]
                    splits = []
                    num_splits = 3888
                    for i in range(num_splits):
                        last = splits[-1][1] if splits else 0
                        splits.append((last, last + HW // num_splits))
                    
                    concat_colors = None
                    for start, end in splits:
                        cam_origins = camera_origin[start:end]
                        ray_directions = ray_dirs[start:end]
                        intervals = torch.linspace(tn, tf, Nc + Nf, device=device).expand(cam_origins.shape[0], -1)
                        ts = ts.reshape(cam_origins.shape[0], Nc+Nf, 1) # size (Batch_size, Nc+Nf, 1)
                        points = cam_origins.unsqueeze(1) + ts * ray_directions.unsqueeze(1)
                        
                        density, out_colors= fine_model(points, ray_directions)
                        pred_colors, _ = integrate_color(Nc + Nf, density, out_colors, ts, device)
                        if concat_colors is None:
                            concat_colors = pred_colors
                        else:
                            concat_colors = torch.concat((concat_colors, pred_colors), dim=0)
                        
                    out_colors = concat_colors.transpose(0,1).reshape(3, -1,  W) #reshape(3, cy*2, cx*2) # pixel colors in [0, 1]

                    # visualize out_colors
                    colors = colors.transpose(0,1).reshape(3, H, W).to(device)
                    grid =  utils.make_grid(torch.cat((out_colors.unsqueeze(0), colors.unsqueeze(0)), dim=0), scale_each=False, normalize=True)
                    #summary_writer.add_image('Ground Truth', colors, global_step=step)
                    summary_writer.add_image('NeRF Output', grid, global_step=step)


if __name__ ==  '__main__':
    train()

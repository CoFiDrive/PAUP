"""Planning-request geometry, independent of any simulator or dataset package."""
import torch
from torch.nn import functional as F
from paup.selection import aligned_request


def gaussian_request(waypoints, height=96, width=288, metres_per_cell=.25,
                     radius=160, sigma_reverse=5.35, sampling_affines=None):
    """Planner lateral/negative-forward points -> supporter-grid request weights.

    `sampling_affines` maps each output supporter grid to the ego input grid,
    exactly the affine_grid/grid_sample convention with align_corners=False.
    Supply already aligned, unexpired points from `aligned_request`.
    """
    points=torch.as_tensor(waypoints)
    if points.ndim!=2 or points.shape[-1]!=2 or not torch.isfinite(points).all():
        raise ValueError('Expected finite K,2 aligned planner waypoints')
    if len(points)==0: raise ValueError('No future request waypoints remain')
    if metres_per_cell<=0 or radius<1 or sigma_reverse<=0:
        raise ValueError('Invalid raster scale')
    # Native waypoint quantization uses the input dtype; Gaussian raster / warp
    # use float64, preserving the original numerical geometry convention.
    centers=torch.stack([points[:,0]/metres_per_cell+height/2,
                         -points[:,1]/metres_per_cell+width/2],1)
    yy=torch.arange(height,device=points.device,dtype=torch.float64)[:,None]
    xx=torch.arange(width,device=points.device,dtype=torch.float64)[None,:]
    raster=torch.zeros((height,width),device=points.device,dtype=torch.float64); sigma=(2*radius+1)/sigma_reverse
    for center in centers:
        row=int(center[0]);col=int(center[1])
        if not (0<=row<height and 0<=col<width): continue
        dx,dy=xx-col,yy-row
        value=torch.exp(-(dx.square()+dy.square())/(2*sigma*sigma))
        value=value*((dx.abs()<=radius)&(dy.abs()<=radius))
        raster=torch.maximum(raster,value)
    if sampling_affines is None:
        warped=raster[None,None]
    else:
        affine=torch.as_tensor(sampling_affines,device=points.device,dtype=raster.dtype)
        source=raster[None,None].expand(len(affine),1,height,width)
        grid=F.affine_grid(affine,source.shape,align_corners=False)
        warped=F.grid_sample(source,grid,align_corners=False).clamp(0,1)
    return (warped*5/(warped.max()+1e-7)).clamp(1e-4,1-1e-4)

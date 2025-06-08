import torch
import numpy as np
from torch.utils.data import Dataset
from utils import *
from itertools import combinations

class EnsumbleParamDataset(Dataset):
    def __init__(self, params:list):
        ''' Input: param 1,2,3 and x,y,z '''
        self.params = params

    def __len__(self):
        return len(self.params)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        return self.params[idx]
    

class ScalarDataset(Dataset):
    def __init__(self, coords:np.array, scalar_field_src:str):
        ''' Input: param 1,2,3 and x,y,z '''
        self.coords = coords
        self.scalar_field = torch.from_numpy(ReadScalarBinary(scalar_field_src))

    def getScalarField(self):
        return self.scalar_field

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.coords[idx], self.scalar_field[idx]

###############################################################################################################

class DecompGrid(torch.nn.Module):
    '''
    grid_shape: [x_3d, y_3d, z_3d, x_2d, y_2d, z_2d, ..._2d]
    '''
    def __init__(self, grid_shape, num_feat_3d, num_feat_2d, num_feat_1d) -> None:
        super().__init__()
        assert num_feat_2d == num_feat_3d
        
        self.grid_shape = grid_shape
        self.num_feat_3d = num_feat_3d
        self.num_feat_2d = num_feat_2d
        self.num_feat_1d = num_feat_1d
        self.feature_grid_3d = torch.nn.Parameter(
            torch.Tensor(1, num_feat_3d, *reversed(grid_shape[:3])),
            requires_grad=True
        )
        torch.nn.init.uniform_(self.feature_grid_3d, a=-0.001, b=0.001)
        
        self.plane_dimid = list(combinations(range(len(grid_shape[3:6])), 2))
        self.plane_dims = list(combinations(grid_shape[3:6], 2))
        self.line_dimid = list(range(3, 3+len(grid_shape[6:])))
        self.line_dims = grid_shape[6:]
        self.planes = []
        self.lines = []
        print('plane dimid', self.plane_dimid)
        print('plane dims', self.plane_dims)
        print('line dimid', self.line_dimid)
        print('line dims', self.line_dims)
        for i, dims in enumerate(self.plane_dims):
            plane = torch.nn.Parameter(
                torch.Tensor(1, num_feat_2d, *reversed(dims)),
                requires_grad=True
            )
            torch.nn.init.uniform_(plane, a=0.999, b=1.001)
            self.planes.append(plane)
        self.planes = torch.nn.ParameterList(self.planes)

        for i, dim in enumerate(self.line_dims):
            line = torch.nn.Parameter(
                torch.Tensor(num_feat_1d, dim),
                requires_grad=True
            )
            torch.nn.init.uniform_(line, a=(0.01)**(1/len(self.line_dimid)), b=(0.02)**(1/len(self.line_dimid)))
            self.lines.append(line)
        self.lines = torch.nn.ParameterList(self.lines)
        
        # initialize with Uniform(-1e-4, 1e-4)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        input: (Batch, Ndim)
        output: (Batch, num_feat_3d/2d)
        '''
        coords = x[..., :3]
        spatial_feats = torch.nn.functional.grid_sample(self.feature_grid_3d,
                            coords.reshape(([1]*coords.shape[-1]) + list(coords.shape)),
                            mode='bilinear', align_corners=True)
        spatial_feats = spatial_feats.squeeze()
        for i, dimids in enumerate(self.plane_dimid):
            x2d = x[:,dimids]
            x2d = x2d.reshape(([1]*x2d.shape[-1]) + list(x2d.shape))
            f2d = torch.nn.functional.grid_sample(self.planes[i],
                            x2d,
                            mode='bilinear', align_corners=True)
            f2d = f2d.squeeze()
            spatial_feats = spatial_feats * f2d
        param_feats = 1.
        for i, dimids in enumerate(self.line_dimid):
            p1d = x[:,dimids]
            p1dn = p1d*(self.line_dims[i]-1)
            p1d_f = torch.floor(p1dn)
            weights = p1dn-p1d_f
            f1d = torch.lerp(self.lines[i][:,p1d_f.type(torch.int)], self.lines[i][:,torch.clamp(p1d_f+1.0, min=0.0, max=self.line_dims[i]-1).type(torch.int)], weights)
            f1d = f1d.squeeze()
            param_feats = param_feats * f1d
        if len(spatial_feats.shape) == 1:
            feats = torch.cat((spatial_feats, param_feats))
            return feats
        feats = torch.cat((spatial_feats.T, param_feats.T), 1)
        return feats

    def forwardWithIntermediates(self, x: torch.Tensor) -> torch.Tensor:
        coords = x[..., :3]
        spatial_feats = torch.nn.functional.grid_sample(self.feature_grid_3d,
                            coords.reshape(([1]*coords.shape[-1]) + list(coords.shape)),
                            mode='bilinear', align_corners=True)
        spatial_feats = spatial_feats.squeeze()
        for i, dimids in enumerate(self.plane_dimid):
            x2d = x[:,dimids]
            x2d = x2d.reshape(([1]*x2d.shape[-1]) + list(x2d.shape))
            f2d = torch.nn.functional.grid_sample(self.planes[i],
                            x2d,
                            mode='bilinear', align_corners=True)
            f2d = f2d.squeeze()
            spatial_feats = spatial_feats * f2d
        param_feats = 1.
        for i, dimids in enumerate(self.line_dimid):
            p1d = x[:,dimids]
            p1dn = p1d*(self.line_dims[i]-1)
            p1d_f = torch.floor(p1dn)
            weights = p1dn-p1d_f
            p1d_f = p1d_f
            f1d = torch.lerp(self.lines[i][:,p1d_f.type(torch.int)], self.lines[i][:,torch.clamp(p1d_f+1.0, min=0.0, max=self.line_dims[i]-1).type(torch.int)], weights)
            f1d = f1d.squeeze()
            param_feats = param_feats * f1d
        if len(spatial_feats.shape) == 1:
            feats = torch.cat((spatial_feats, param_feats))
            return feats, spatial_feats, param_feats
        feats = torch.cat((spatial_feats.T, param_feats.T), 1)
        return feats, spatial_feats.T, param_feats.T
    
    def piecewise_linear_mean_var_torch_multi(self, lineidx:int, values:torch.Tensor, xrange:torch.Tensor):
        # precompute means for each piece
        means = (values[:,:-1] + values[:,1:]) * 0.5
        # function to compute variance on a piece
        def contfunc_var(xleft, xright, m):
            return (xleft-m)*(xright-m) + ((xright-xleft)**2) / 3
        # find bin index for xmin and xmax, compute weights for linear interpolation
        xf = torch.floor(xrange*(self.line_dims[lineidx]-1))
        xf_i = xf.type(torch.long)
        weights = xrange*(self.line_dims[lineidx]-1) - xf
        weights = weights.to('cuda:0')
        xrange2y = torch.lerp(values[:,xf_i], values[:,xf_i+1], weights)
        # xrange may contains complete piece
        if xf_i[0] != xf_i[1]:
            headmean = (xrange2y[:,0] + values[:,xf_i[0]+1])*0.5
            tailmean = (xrange2y[:,1] + values[:,xf_i[1]])*0.5
            weighted_sum = headmean * (1.0-weights[0]) + tailmean * weights[1] + torch.sum(means[:,xf_i[0]+1: xf_i[1]], 1)
            range_query_mean = weighted_sum / ((xrange[1]-xrange[0])*(self.line_dims[lineidx]-1))
            range_query_var = 0.0
            range_query_var += contfunc_var(xrange2y[:,0], values[:,xf_i[0]+1], range_query_mean) * (1.0-weights[0])
            range_query_var += contfunc_var(xrange2y[:,1], values[:,xf_i[1]], range_query_mean) * weights[1]
            for i in range(xf_i[0]+1, xf_i[1]):
                range_query_var += contfunc_var(values[:,i], values[:,i+1], range_query_mean)
            range_query_var /= ((xrange[1]-xrange[0])*(self.line_dims[lineidx]-1))
        else:
            range_query_mean = (xrange2y[:,0] + xrange2y[:,1])*0.5
            range_query_var = contfunc_var(xrange2y[:,0], xrange2y[:,1], range_query_mean)
        return range_query_mean, range_query_var

    def param_range_query(self, coords: torch.Tensor, pmin: torch.Tensor, pmax: torch.Tensor) -> torch.Tensor:
        assert len(pmin) == len(pmax)
        # compute spatial features (fixed)
        spatial_feats = torch.nn.functional.grid_sample(self.feature_grid_3d,
                            coords.reshape(([1]*coords.shape[-1]) + list(coords.shape)),
                            mode='bilinear', align_corners=True)
        spatial_feats = spatial_feats.squeeze()
        for i, dimids in enumerate(self.plane_dimid):
            x2d = coords[:,dimids]
            x2d = x2d.reshape(([1]*x2d.shape[-1]) + list(x2d.shape))
            f2d = torch.nn.functional.grid_sample(self.planes[i],
                            x2d,
                            mode='bilinear', align_corners=True)
            f2d = f2d.squeeze()
            spatial_feats = spatial_feats * f2d
        param_feature_mean, var_plus_mean, mean_sq = 1., 1., 1.
        for i, dimids in enumerate(self.line_dimid):
            prange = torch.Tensor([pmin[i], pmax[i]])
            curmean, curvar = self.piecewise_linear_mean_var_torch_multi(i, self.lines[i], prange)
            print('curmean', curmean)
            print('curvar', curvar)
            param_feature_mean = param_feature_mean*curmean
            var_plus_mean = var_plus_mean * (curvar + curmean**2)
            mean_sq = mean_sq * (curmean**2)
        param_feature_var = var_plus_mean - mean_sq
        return spatial_feats, param_feature_mean, param_feature_var
    
    def param_range_query_by_sample(self, coords: torch.Tensor, pmin: torch.Tensor, pmax: torch.Tensor) -> torch.Tensor:
        assert len(pmin) == len(pmax)
        # compute spatial features (fixed)
        spatial_feats = torch.nn.functional.grid_sample(self.feature_grid_3d,
                            coords.reshape(([1]*coords.shape[-1]) + list(coords.shape)),
                            mode='bilinear', align_corners=True)
        spatial_feats = spatial_feats.squeeze()
        for i, dimids in enumerate(self.plane_dimid):
            x2d = coords[:,dimids]
            x2d = x2d.reshape(([1]*x2d.shape[-1]) + list(x2d.shape))
            f2d = torch.nn.functional.grid_sample(self.planes[i],
                            x2d,
                            mode='bilinear', align_corners=True)
            f2d = f2d.squeeze()
            spatial_feats = spatial_feats * f2d
        # compute param_features by sample
        def piecewise_query(values, x, lineidx):
            xf = torch.floor(x*(self.line_dims[lineidx]-1))
            xf_i = xf.type(torch.long)
            w = x*(self.line_dims[lineidx]-1) - xf
            w = w.to('cuda:0')
            return torch.lerp(values[:,xf_i], values[:,xf_i+1], w)
        ysmul = None
        for i, dimids in enumerate(self.line_dimid):
            cur_xs = torch.linspace(pmin[i], pmax[i], 200)
            cur_ys = piecewise_query(values=self.lines[i], x=cur_xs, lineidx=i)
            print('cur_ys_mean', torch.mean(cur_ys, 1))
            print('cur_ys_var', torch.var(cur_ys, 1))
            if ysmul is None:
                ysmul = cur_ys
            else:
                new_ysmul = None
                for i in range(cur_ys.shape[1]):
                    if new_ysmul is None:
                        new_ysmul = cur_ys[:,i][:,None] * ysmul
                    else:
                        new_ysmul = torch.cat((new_ysmul, cur_ys[:,i][:,None] * ysmul), 1)
                ysmul = new_ysmul
            
        return spatial_feats, torch.mean(ysmul, 1), torch.var(ysmul, 1)
        
    
###############################################################################################################
    
class INR_FG(torch.nn.Module):
    def __init__(self, grid_shape, num_feat_3d, num_feat_2d, num_feat_1d, out_features:int) -> None:
        super().__init__()
        self.dg = DecompGrid(grid_shape=grid_shape, num_feat_3d=num_feat_3d, num_feat_2d=num_feat_2d, num_feat_1d=num_feat_1d)
        
        self.hidden_nodes = 128
        self.fc1 = torch.nn.Linear(num_feat_3d+num_feat_1d, self.hidden_nodes)
        self.fc2 = torch.nn.Linear(self.hidden_nodes, self.hidden_nodes)
        self.fc3 = torch.nn.Linear(self.hidden_nodes, self.hidden_nodes)
        self.fc4 = torch.nn.Linear(self.hidden_nodes, out_features)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.normal_(self.fc1.bias, 0, 0.001)
        torch.nn.init.xavier_normal_(self.fc2.weight)
        torch.nn.init.normal_(self.fc2.bias, 0, 0.001)
        torch.nn.init.xavier_normal_(self.fc3.weight)
        torch.nn.init.normal_(self.fc3.bias, 0, 0.001)
        torch.nn.init.xavier_normal_(self.fc4.weight)
        torch.nn.init.normal_(self.fc4.bias, 0, 0.001)

    def forward(self, x):
        x = self.dg(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x
    
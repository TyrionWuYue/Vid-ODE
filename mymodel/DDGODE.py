import torch
import torch.nn as nn
from mymodel.DyGCNCell import DyGCRNCell
from mymodel.ODE import ODEBlock
from mymodel.ConvLayers import *


class DDGODE(nn.Module):
    def __init__(self, opt, device):
        super(DDGODE, self).__init__()

        self.opt = opt
        self.device = device
        self.frame_channel = opt.input_dim
        self.init_dim = opt.init_dim
        self.n_downs = opt.n_downs
        self.num_layers = 1
        self.num_hidden = 128
        resize = 2 ** self.n_downs
        self.base_dim = self.init_dim * resize
        self.sample_size = opt.sample_size

        self.width = opt.input_size
        self.node_num = (self.width // resize) ** 2

        self.encoder = Encoder(self.frame_channel, self.init_dim, n_downs=self.n_downs) # Encoder
        self.decoder = Decoder(self.base_dim*2, self.frame_channel+3, n_ups=self.n_downs)
        cell_list = []
        for i in range(self.num_layers):
            cell_list.append(
                DyGCRNCell(node_num=self.node_num, dim_in=self.base_dim, hidden_dim=self.base_dim, mem_num=32, embed_dim=10, width=self.width // resize)
            )
        self.cell_list = nn.ModuleList(cell_list)

        dec_cell_list = []
        for i in range(self.num_layers):
            cell_list.append(
                DyGCRNCell(node_num=self.node_num, dim_in=self.base_dim, hidden_dim=self.base_dim, mem_num=32, embed_dim=10, width=self.width // resize)
            )
        self.dec_cell_list = nn.ModuleList(dec_cell_list)
        
        self.odeblock = ODEBlock(self.base_dim, self.base_dim*2, self.sample_size//2)
    

    def get_reconstruction(self, time_steps_to_predict, truth, truth_time_steps, mask=None, out_mask=None):
        truth = truth.to(self.device)
        truth_time_steps = truth_time_steps.to(self.device)
        time_steps_to_predict = time_steps_to_predict.to(self.device)
        resize = 2 ** self.n_downs
        b, t, c, h, w = truth.shape
        pred_t_len = len(time_steps_to_predict)

        # Skip Connection
        skip_image = truth[:, -1, ...]
        skip_conn_embed = self.encoder(skip_image).view(b, -1, h // resize, w // resize)

        # Conv encoding
        e_truth = self.encoder(truth.view(b*t, c, h, w)).view(b, t, -1, h // resize, w // resize)
        e_gen = self.odeblock(e_truth[:, -1, ...])

        # DyGCN
        h_t = []
        for i in range(self.num_layers):
            zeros = torch.zeros(b, self.base_dim, h // resize, w // resize).to(self.device)
            h_t.append(zeros)
        
        for t in range(len(truth_time_steps)):
            h_t[0] = self.cell_list[0](e_truth[:, t, ...], h_t[0])
            for i in range(1, self.num_layers):
                h_t[i] = self.cell_list[i](h_t[i-1], h_t[i])
        
        first_point_enc = h_t[-1]

        sol_y = []
        for i in range(pred_t_len):
            h_t[0] = self.cell_list[0](e_gen[:, i, ...], h_t[0])
            for j in range(1, self.num_layers):
                h_t[j] = self.cell_list[j](h_t[j-1], h_t[j])
            sol_y.append(h_t[-1])
        sol_y = torch.stack(sol_y, dim=1)
        sol_y = sol_y.contiguous().view(b, pred_t_len, self.base_dim, h // resize, w // resize)
        
        # regular b, t, 6, h, w / irregular b, t * ratio, 6, h, w
        pred_outputs = self.get_flowmaps(sol_out=sol_y, first_prev_embed=skip_conn_embed, mask=out_mask) # b, t, 6, h, w
        pred_outputs = torch.cat(pred_outputs, dim=1)
        pred_flows, pred_intermediates, pred_masks = \
            pred_outputs[:, :, :2, ...], pred_outputs[:, :, 2:2+self.opt.input_dim, ...], torch.sigmoid(pred_outputs[:, :, 2+self.opt.input_dim:, ...])

        ### Warping first frame by using optical flow
        # Declare grid for warping
        grid_x = torch.linspace(-1.0, 1.0, w).view(1, 1, w, 1).expand(b, h, -1, -1)
        grid_y = torch.linspace(-1.0, 1.0, h).view(1, h, 1, 1).expand(b, -1, w, -1)
        grid = torch.cat([grid_x, grid_y], 3).float().to(self.device)  # [b, h, w, 2]

        # Warping
        last_frame = truth[:, -1, ...]
        warped_pred_x = self.get_warped_images(pred_flows=pred_flows, start_image=last_frame, grid=grid)
        warped_pred_x = torch.cat(warped_pred_x, dim=1)  # regular b, t, 6, h, w / irregular b, t * ratio, 6, h, w

        pred_x = pred_masks * warped_pred_x + (1 - pred_masks) * pred_intermediates
        
        pred_x = pred_x.view(b, -1, c, h, w)
        
        ### extra information
        extra_info = {}
        
        extra_info["optical_flow"] = pred_flows
        extra_info["warped_pred_x"] = warped_pred_x
        extra_info["pred_intermediates"] = pred_intermediates
        extra_info["pred_masks"] = pred_masks
        
        return pred_x, extra_info

    def get_mse(self, truth, pred_x, mask=None):
    
        b, _, c, h, w = truth.size()
        
        if mask is None:
            selected_time_len = truth.size(1)
            selected_truth = truth
        else:
            selected_time_len = int(mask[0].sum())
            selected_truth = truth[mask.squeeze(-1).bool()].view(b, selected_time_len, c, h, w)
        loss = torch.sum(torch.abs(pred_x - selected_truth)) / (b * selected_time_len * c * h * w)
        return loss
    
    
    def get_diff(self, data, mask=None):
        
        data_diff = data[:, 1:, ...] - data[:, :-1, ...]
        b, _, c, h, w = data_diff.size()
        selected_time_len = int(mask[0].sum())
        masked_data_diff = data_diff[mask.squeeze(-1).bool()].view(b, selected_time_len, c, h, w)
        
        return masked_data_diff

    
    def export_infos(self):
        infos = self.tracker.export_info()
        self.tracker.clean_info()
        return infos
    
    def get_flowmaps(self, sol_out, first_prev_embed, mask):
        """ Get flowmaps recursively
        Input:
            sol_out - Latents from ODE decoder solver (b, time_steps_to_predict, c, h, w)
            first_prev_embed - Latents of last frame (b, c, h, w)
        
        Output:
            pred_flows - List of predicted flowmaps (b, time_steps_to_predict, c, h, w)
        """
        b, _, c, h, w = sol_out.size()
        pred_time_steps = int(mask[0].sum())
        pred_flows = list()
    
        prev = first_prev_embed.clone()
        time_iter = range(pred_time_steps)
        
        if mask.size(1) == sol_out.size(1):
            sol_out = sol_out[mask.squeeze(-1).bool()].view(b, pred_time_steps, c, h, w)
        
        for t in time_iter:
            cur_and_prev = torch.cat([sol_out[:, t, ...], prev], dim=1)
            pred_flow = self.decoder(cur_and_prev).unsqueeze(1)
            pred_flows += [pred_flow]
            prev = sol_out[:, t, ...].clone()
    
        return pred_flows
    
    def get_warped_images(self, pred_flows, start_image, grid):
        """ Get warped images recursively
        Input:
            pred_flows - Predicted flowmaps to use (b, time_steps_to_predict, c, h, w)
            start_image- Start image to warp
            grid - pre-defined grid

        Output:
            pred_x - List of warped (b, time_steps_to_predict, c, h, w)
        """
        warped_time_steps = pred_flows.size(1)
        pred_x = list()
        last_frame = start_image
        b, _, c, h, w = pred_flows.shape
        
        for t in range(warped_time_steps):
            pred_flow = pred_flows[:, t, ...]           # b, 2, h, w
            pred_flow = torch.cat([pred_flow[:, 0:1, :, :] / ((w - 1.0) / 2.0), pred_flow[:, 1:2, :, :] / ((h - 1.0) / 2.0)], dim=1)
            pred_flow = pred_flow.permute(0, 2, 3, 1)   # b, h, w, 2
            flow_grid = grid.clone() + pred_flow.clone()# b, h, w, 2
            warped_x = nn.functional.grid_sample(last_frame, flow_grid, padding_mode="border", align_corners=False)
            pred_x += [warped_x.unsqueeze(1)]           # b, 1, 3, h, w
            last_frame = warped_x.clone()
        
        return pred_x
    
    def compute_all_losses(self, batch_dict):
        
        batch_dict["tp_to_predict"] = batch_dict["tp_to_predict"].to(self.device)
        batch_dict["observed_data"] = batch_dict["observed_data"].to(self.device)
        batch_dict["observed_tp"] = batch_dict["observed_tp"].to(self.device)
        batch_dict["observed_mask"] = batch_dict["observed_mask"].to(self.device)
        batch_dict["data_to_predict"] = batch_dict["data_to_predict"].to(self.device)
        batch_dict["mask_predicted_data"] = batch_dict["mask_predicted_data"].to(self.device)

        pred_x, extra_info = self.get_reconstruction(
            time_steps_to_predict=batch_dict["tp_to_predict"],
            truth=batch_dict["observed_data"],
            truth_time_steps=batch_dict["observed_tp"],
            mask=batch_dict["observed_mask"],
            out_mask=batch_dict["mask_predicted_data"])
        
        # batch-wise mean
        loss = torch.mean(self.get_mse(truth=batch_dict["data_to_predict"],
                                       pred_x=pred_x,
                                       mask=batch_dict["mask_predicted_data"]))

        init_image = batch_dict["observed_data"][:, -1, ...]

        data = torch.cat([init_image.unsqueeze(1), batch_dict["data_to_predict"]], dim=1)
        data_diff = self.get_diff(data=data, mask=batch_dict["mask_predicted_data"])

        loss = loss + torch.mean(self.get_mse(truth=data_diff, pred_x=extra_info["pred_intermediates"], mask=None))

        results = {}
        results["loss"] = torch.mean(loss)
        results["pred_y"] = pred_x
        
        return results
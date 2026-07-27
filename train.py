# JLSAE: Joint Low-Rank and Sparse Autoencoder for Hyperspectral Anomaly Detection
#
# Place the test data in MATLAB format (.mat) in the "data" folder.
# Each .mat file should contain two variables:
#   - "data": the hyperspectral image
#   - "map":  the ground-truth anomaly map
#
# The detection results will be saved in the "result" folder as a .mat file,
# containing:
#   - "detection": the generated anomaly detection map
#   - "auc":       the area under the ROC curve
#
# Note that the network parameters and the input noise Z are randomly
# initialized for each run. Therefore, the detection results and AUC values
# may vary slightly across different runs.

from models.skip import skip
from utils.common_utils import *
from utils.draw_ROC_utils import *
import numpy as np
import torch
import os
from scipy import io

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor

# load data Y
path = 'data/'
file_name = 'HYDICE.mat'
Y_np, y_true = load(path, file_name)
Y_tensor = torch.tensor(Y_np).type(dtype).detach()
Y_tensor = Y_tensor.unsqueeze(0)

# create input Z
shape = Y_tensor.shape
net_input = torch.zeros(shape)
net_input.uniform_()
net_input *= 1./10
net_input = net_input.type(dtype).detach()


Y_np = Y_np.transpose(1, 2, 0)

layer_num = 5
net = skip(
    shape[1], shape[1],
    num_channels_down=[128] * layer_num,
    num_channels_up=[128] * layer_num,
    num_channels_skip=[128] * layer_num,
    upsample_mode='bilinear',
    need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU')

net = net.type(dtype)

s = sum([np.prod(list(p.size())) for p in net.parameters()])
print('Number of params: %d' % s)

mse = torch.nn.MSELoss().type(dtype)

i = 0
reg_noise_std = 0  # 0 0.01 0.03 0.05
decay = True
LR = 0.01

pho = 1  #Try 1.25, 1.5, or 1.75 if the result is unsatisfactory. For Pavia, set pho = 1.75
lam = 0.5

OPTIMIZER = 'adam'  # 'LBFGS'
OPT_OVER = 'net'
if file_name == 'Bay Champagne.mat':
    num_iter = 100
elif file_name == 'HYDICE.mat':
    num_iter = 300
elif file_name == 'Pavia.mat':
    num_iter = 170
elif file_name == 'Hyperion.mat':
    num_iter = 150
else:
    # Default setting for other datasets.
    # It can be tuned within [100, 150, 200, 250, 300] for potentially better performance.
    num_iter = 150
out = torch.zeros_like(Y_tensor).type(dtype).detach()
X_k = torch.zeros_like(Y_tensor).type(dtype).detach()
U1_k = torch.zeros_like(Y_tensor).type(dtype).detach()
U2_k = torch.zeros_like(Y_tensor).type(dtype).detach()
E_k = torch.zeros_like(Y_tensor).type(dtype).detach()
mask_var = torch.ones_like(Y_tensor).type(dtype).detach()

net_input_saved = net_input.detach().clone()
noise = net_input.detach().clone()

def closure():
    global i, net_input, out, X_k, U1_k, U2_k, E_k
    X_k = D(out, U2_k, pho)

    net_input = net_input_saved
    if reg_noise_std > 0:
        net_input = net_input_saved + (noise.normal_() * reg_noise_std)

    out = net(net_input)
    if i!=0:
        net_output_clone = out.detach().clone()
        temp = (net_output_clone[0, :] - Y_tensor[0, :]) * (net_output_clone[0, :] - Y_tensor[0, :])
        residual_img = temp.sum(0)
        r_max = residual_img.max()
        residual_img = r_max - residual_img
        r_min, r_max = residual_img.min(), residual_img.max()
        residual_img = (residual_img - r_min) / (r_max - r_min)
        mask_size = mask_var.size()
        for j in range(mask_size[1]):
            mask_var[0, j, :] = residual_img[:]

    total_loss = pho/2*(mse(Y_tensor * mask_var, (out + E_k + U1_k/pho) * mask_var) + mse(out * mask_var, (X_k + U2_k/pho) * mask_var))
    total_loss.backward()
    print('iter:', i, 'loss:', total_loss.item())
    E_k = M(Y_tensor, out, U1_k, pho, lam)
    U1_k = (U1_k + pho*(out + E_k - Y_tensor)).detach()
    U2_k = (U2_k + pho*(X_k - out)).detach()
    i += 1
    return total_loss


p = get_params(OPT_OVER, net, net_input)
optimize(OPTIMIZER, p, closure, LR, num_iter)

out = out.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
X_k = X_k.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
E_k = E_k.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)

# In most cases, the L2 norm above is sufficient. For a few data sets, the Mahalanobis distance may provide better detection performance.
# If you would like to evaluate the performance of the Mahalanobis distance, you can use the commented-out code in the 'construction_error' function in draw_ROC_utils.py.
y_score = construction_error(Y_np, out)

# Gamma transformation: gamma > 1 suppresses background responses but may also weaken anomaly responses,
# whereas gamma < 1 enhances anomaly responses at the cost of potentially increasing false alarms.
# gamma = 1.1
# y_score = np.power(np.clip(y_score, 0, 1), gamma)

fpr, tpr, thresholds = calculate_fpr_and_fpr(y_true, y_score)
auc = calculate_auc(fpr, tpr, method='trapezoid')

dataset_name = os.path.splitext(os.path.basename(file_name))[0]
result_path = os.path.join('result', f'{dataset_name}_result.mat')
io.savemat(result_path, {'auc': auc, 'detection': y_score})

draw_roc(fpr, tpr)
print(calculate_auc(fpr, tpr, method='trapezoid'))


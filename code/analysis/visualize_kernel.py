proj_dir = '/Users/minzhe.zhang/Documents/DeepLearning_LTV'
import os
import sys
sys.path.append(os.path.join(proj_dir, 'code'))
import numpy as np
import matplotlib.pyplot as plt
from model.neural_net_keras import cnn_lstm

################            funcrtion            ###################
def plot_kernel(model, kernel_name, n_channel, layer_dim, fig_path):
    row = int(np.ceil(n_channel/8))
    fig, ax = plt.subplots(row, 8, sharex=True, sharey='row', figsize=(16, row))
    for i in range(n_channel):
        print(i)
        ts = model.kernel_vis(kernel_name, kernel_index=i, input_index=1, layer_dim=layer_dim)
        # if (abs(ts) < 5).all():
        #     ax[i//8, i%8].set_ylim((-5,5))
        ax[i//8, i%8].plot(ts)
    fig.savefig(fig_path)

### -------------------  visualize kernel  -------------------- ###
kernel_name = 'conv1d_3'
fig_path = os.path.join(proj_dir, 'result/img/kernel_vis.{}.png'.format(kernel_name))
model_path = os.path.join(proj_dir, 'code/model/archive/cnn_lstm.52_52@in.(52,52,52).2.83.19-out.1-conv.48.4.None.conv.96.4.4.conv.172.3.3-lstm.128.0.0-fc.320.0.2.fc.256.0.1-loss.wmse-batch.128/model.pkl')
model = cnn_lstm.load(model_path)
model.load_weights()
# plot_kernel(model, kernel_name, n_channel=42, layer_dim=3, fig_path=fig_path)
ts = model.kernel_vis(kernel_name, kernel_index=slice(None,None,None), input_index=0, layer_dim=3)
plt.plot(ts)
plt.show()
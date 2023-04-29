import pandas as pd
import numpy as np

n_dim = 15

chains = np.load('./result/GW150914.npz')['chains']

samples_all = chains.reshape(-1,n_dim)

labels = ['$M_c$', '$q$', '$\chi_1$', '$\chi_2$', '$d_L$', '$t_c$', '$\phi_c$', '$\cos\iota$', '$\psi$', '$RA$', '$\sin({DEC})$',
          '$d_{L2}$', '$t_2$', '$n_1$', '$n_2$'
          ]

df = pd.DataFrame()
for i in range(n_dim):
    df[labels[i]] = samples_all[:,i]

for i in [0, 0.5, 1]:
    mask1 = (df['$n_1$'] >= i) & (df['$n_1$'] < i + 0.5)
    df.loc[mask1, '$n_1$'] = i
    mask2 = (df['$n_2$'] >= i) & (df['$n_2$'] < i + 0.5)
    df.loc[mask2, '$n_2$'] = i

import seaborn as sns

g = sns.pairplot(df,
                 vars = ['$M_c$', '$q$', '$\chi_1$', '$\chi_2$', '$d_L$', '$t_c$', '$\phi_c$', '$\cos\iota$', '$\psi$', '$RA$', '$\sin({DEC})$',
                        #  '$d_{L2}$', '$t_2$', '$n_1$', '$n_2$'
                         ],
                 corner=True, kind='hist',
                 diag_kws=dict(common_norm=False),
                 plot_kws=dict(common_norm=False, rasterized=True))

# from bilby.gw.result import CBCResult
# result_GR = CBCResult.from_json("./data/GW150914_GR.json.gz").posterior
# result_GR['cos_iota'] = np.cos([float(value) for value in result_GR['iota']])

# trigger_time = 1126259462.4

# true_param = np.array([result_GR['chirp_mass'].median(),
#                        result_GR['mass_ratio'].median(),
#                        result_GR['a_1'].median(),
#                        result_GR['a_2'].median(),
#                        result_GR['luminosity_distance'].median(),
#                        result_GR['geocent_time'].median() - trigger_time,
#                        result_GR['phase'].median(),
#                        result_GR['cos_iota'].median(),
#                        result_GR['psi'].median(),
#                        result_GR['ra'].median(),
#                        np.sin(result_GR['dec']).median()])

# for i in range(11):
#     g.axes[i,i].axvline(true_param[i], color=sns.color_palette()[3])
#     for j in range(i):
#         g.axes[i,j].axvline(true_param[j], color=sns.color_palette()[3])
#         g.axes[i,j].axhline(true_param[i], color=sns.color_palette()[3])

g.figure.savefig('./result/corner.pdf')

import pandas as pd
import numpy as np

n_dim = 11

chains = np.load('./result/result.npz')['chains']
injection_parameters = np.load('./result/result.npz', allow_pickle=True)['injection_parameters'].item()

samples_all = chains.reshape(-1,n_dim)

labels = ['$M_c$', '$q$', '$\chi_1$', '$\chi_2$', '$d_L$', '$t_c$', '$\phi_c$', '$\cos\iota$', '$\psi$', '$RA$', '$\sin({DEC})$']

df = pd.DataFrame()
for i in range(11):
    df[labels[i]] = samples_all[:,i]

df = df.sample(n=50000)

import seaborn as sns

g = sns.pairplot(df, corner=True, kind='hist',
                 diag_kws=dict(common_norm=False, rasterized=True),
                 plot_kws=dict(common_norm=False))

trigger_time = 1126259542.9

from bilby.gw.conversion import component_masses_to_chirp_mass, component_masses_to_mass_ratio

true_param = np.array([component_masses_to_chirp_mass(injection_parameters['mass_1'], injection_parameters['mass_2']),
                       component_masses_to_mass_ratio(injection_parameters['mass_1'], injection_parameters['mass_2']),
                       injection_parameters['chi_1'],
                       injection_parameters['chi_2'],
                       injection_parameters['luminosity_distance'],
                       injection_parameters['geocent_time'] - trigger_time,
                       injection_parameters['phase'],
                       np.cos(injection_parameters['theta_jn']),
                       injection_parameters['psi'],
                       injection_parameters['ra'],
                       np.sin(injection_parameters['dec'])])

for i in range(n_dim):
    g.axes[i,i].axvline(true_param[i], color=sns.color_palette()[3])
    for j in range(i):
        g.axes[i,j].axvline(true_param[j], color=sns.color_palette()[3])
        g.axes[i,j].axhline(true_param[i], color=sns.color_palette()[3])

g.figure.savefig('./result/corner.pdf')

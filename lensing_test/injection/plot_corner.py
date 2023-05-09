import pandas as pd
import numpy as np

n_dim = 15

chains = np.load('./result/result.npz')['chains']

samples_all = chains.reshape(-1,n_dim)

labels = ['$M_c$', '$q$', '$\chi_1$', '$\chi_2$', '$d_L$', '$t_c$', '$\phi_c$', '$\cos\iota$', '$\psi$', '$RA$', '$\sin({DEC})$',
          '$d_{L2}$', '$t_2$', '$n_1$', '$n_2$'
          ]

df = pd.DataFrame()
for i in range(n_dim):
    df[labels[i]] = samples_all[:,i]

df = df.sample(n=50000)

for i in [0, 0.5, 1]:
    mask1 = (df['$n_1$'] >= i) & (df['$n_1$'] < i + 0.5)
    df.loc[mask1, '$n_1$'] = i
    mask2 = (df['$n_2$'] >= i) & (df['$n_2$'] < i + 0.5)
    df.loc[mask2, '$n_2$'] = i

import seaborn as sns

g = sns.pairplot(df, corner=True, kind='hist',
                 diag_kws=dict(common_norm=False, rasterized=True),
                 plot_kws=dict(common_norm=False))

trigger_time = 1126259642.4

injection_parameters = dict(
    mass_1=36.0,
    mass_2=29.0,
    chi_1=0.4,
    chi_2=0.3,
    luminosity_distance=1000.0,
    theta_jn=0.4,
    psi=2.659,
    phase=1.3,
    geocent_time=1126259642.413,
    ra=1.375,
    dec=-1.2108,
    d_L2=1500,
    dt=1e-3,
    n_1=0,
    n_2=0.5
)

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
                       np.sin(injection_parameters['dec']),
                       injection_parameters['d_L2'],
                       injection_parameters['dt'],
                       injection_parameters['n_1'],
                       injection_parameters['n_2']])

for i in range(n_dim):
    g.axes[i,i].axvline(true_param[i], color=sns.color_palette()[3])
    for j in range(i):
        g.axes[i,j].axvline(true_param[j], color=sns.color_palette()[3])
        g.axes[i,j].axhline(true_param[i], color=sns.color_palette()[3])

g.figure.savefig('./result/corner.pdf')

import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".8"

import numpy as np
import jax.numpy as jnp
import jax
from lal import GreenwichMeanSiderealTime
from gwpy.timeseries import TimeSeries
import bilby
from gwosc.datasets import event_detectors

from ripple.waveforms.IMRPhenomD import gen_IMRPhenomD_polar
from jimgw.PE.detector_preset import * 
# from jimgw.PE.heterodyneLikelihood import make_heterodyne_likelihood
from jimgw.PE.detector_projection import make_detector_response

from flowMC.nfmodel.rqSpline import RQSpline
from flowMC.sampler.Sampler import Sampler
from flowMC.sampler.MALA import MALA
from flowMC.utils.PRNG_keys import initialize_rng_keys
from flowMC.nfmodel.utils import *

event = 'GW150914'

minimum_frequency = 20
maximum_frequency = 1024

trigger_time = 1126259462.4
duration = 4
post_trigger_duration = 2
epoch = duration - post_trigger_duration
gmst = GreenwichMeanSiderealTime(trigger_time)
f_ref = 20
f_sample = 4096

detectors=event_detectors(event)
ifos = bilby.gw.detector.InterferometerList(detectors)

for detector in ifos:
    analysis_data = TimeSeries.fetch_open_data(detector.name, trigger_time-duration+post_trigger_duration, trigger_time+post_trigger_duration, sample_rate=f_sample, cache=True)
    detector.set_strain_data_from_gwpy_timeseries(analysis_data)

H1_frequency = ifos[1].frequency_array
H1_data = ifos[1].frequency_domain_strain
H1_psd_frequency, H1_psd_temp = np.genfromtxt('/home/jason/thomas_folder/project/millilensing/psd/GW150914_psd_H1.dat').T
if H1_psd_frequency[1] - H1_psd_frequency[0] == H1_frequency[1] - H1_frequency[0]:
    H1_psd = np.full(len(H1_frequency), np.inf)
    for i in range(len(H1_psd_frequency)):
        H1_psd[i] = H1_psd_temp[i]
else:
    print('df of H1 PSD is not equal to df of H1 data')

H1_data = H1_data[(H1_frequency>minimum_frequency)*(H1_frequency<maximum_frequency)]
H1_psd = H1_psd[(H1_frequency>minimum_frequency)*(H1_frequency<maximum_frequency)]
H1_frequency = H1_frequency[(H1_frequency>minimum_frequency)*(H1_frequency<maximum_frequency)]

L1_frequency = ifos[0].frequency_array
L1_data = ifos[0].frequency_domain_strain
L1_psd_frequency, L1_psd_temp = np.genfromtxt('/home/jason/thomas_folder/project/millilensing/psd/GW150914_psd_L1.dat').T
if L1_psd_frequency[1] - L1_psd_frequency[0] == L1_frequency[1] - L1_frequency[0]:
    L1_psd = np.full(len(L1_frequency), np.inf)
    for i in range(len(L1_psd_frequency)):
        L1_psd[i] = L1_psd_temp[i]
else:
    print('df of L1 PSD is not equal to df of L1 data')

L1_data = L1_data[(L1_frequency>minimum_frequency)*(L1_frequency<maximum_frequency)]
L1_psd = L1_psd[(L1_frequency>minimum_frequency)*(L1_frequency<maximum_frequency)]
L1_frequency = L1_frequency[(L1_frequency>minimum_frequency)*(L1_frequency<maximum_frequency)]

H1 = get_H1()
H1_response = make_detector_response(H1[0], H1[1])
L1 = get_L1()
L1_response = make_detector_response(L1[0], L1[1])


# def gen_waveform_H1(f, theta, epoch, gmst, f_ref):
#     theta_waveform = theta[:8]
#     theta_waveform = theta_waveform.at[5].set(0)
#     ra = theta[9]
#     dec = theta[10]
#     hp, hc = gen_IMRPhenomD_polar(f, theta_waveform, f_ref)
#     return H1_response(f, hp, hc, ra, dec, gmst , theta[8]) * jnp.exp(-1j*2*jnp.pi*f*(epoch+theta[5]))

# def gen_waveform_L1(f, theta, epoch, gmst, f_ref):
#     theta_waveform = theta[:8]
#     theta_waveform = theta_waveform.at[5].set(0)
#     ra = theta[9]
#     dec = theta[10]
#     hp, hc = gen_IMRPhenomD_polar(f, theta_waveform, f_ref)
#     return L1_response(f, hp, hc, ra, dec, gmst, theta[8]) * jnp.exp(-1j*2*jnp.pi*f*(epoch+theta[5]))

def gen_lensed_IMRPhenomD_polar(f, theta, f_ref):
    hp, hc = gen_IMRPhenomD_polar(f, theta, f_ref)

    def Discrete(x, d):
        return jnp.floor(x/d)*d

    # Amplification function
    w = 1j*2*jnp.pi*f
    F = jnp.exp(-1j*Discrete(theta[13], 0.5)*jnp.pi) + (theta[4]/theta[11])*(jnp.exp(w*theta[12]-1j*Discrete(theta[14], 0.5)*jnp.pi))

    return hp*F, hc*F

def LogLikelihood(theta):
    theta_waveform = theta[:8]
    theta_waveform = theta_waveform.at[5].set(0)
    ra = theta[9]
    dec = theta[10]
    hp_test, hc_test = gen_lensed_IMRPhenomD_polar(H1_frequency, theta_waveform, f_ref)
    align_time = jnp.exp(-1j*2*jnp.pi*H1_frequency*(epoch+theta[5]))
    h_test_H1 = H1_response(H1_frequency, hp_test, hc_test, ra, dec, gmst, theta[8]) * align_time
    h_test_L1 = L1_response(L1_frequency, hp_test, hc_test, ra, dec, gmst, theta[8]) * align_time
    df = H1_frequency[1] - H1_frequency[0]
    match_filter_SNR_H1 = 4*jnp.sum((jnp.conj(h_test_H1)*H1_data)/H1_psd*df).real
    match_filter_SNR_L1 = 4*jnp.sum((jnp.conj(h_test_L1)*L1_data)/L1_psd*df).real
    optimal_SNR_H1 = 4*jnp.sum((jnp.conj(h_test_H1)*h_test_H1)/H1_psd*df).real
    optimal_SNR_L1 = 4*jnp.sum((jnp.conj(h_test_L1)*h_test_L1)/L1_psd*df).real

    return (match_filter_SNR_H1-optimal_SNR_H1/2) + (match_filter_SNR_L1-optimal_SNR_L1/2)


# ref_param = jnp.array([ 3.10497857e+01,  2.46759666e-01,  3.04854781e-01, -4.92774588e-01,
#         5.47223231e+02,  1.29378808e-02,  3.30994042e+00,  3.88802965e-01,
#         3.41074151e-02,  2.55345319e+00, -9.52109059e-01, 6e+02, 1e-3, 0, 5e-1])

# from jimgw.PE.heterodyneLikelihood import make_heterodyne_likelihood_mutliple_detector

# data_list = [H1_data, L1_data]
# psd_list = [H1_psd, L1_psd]
# response_list = [H1_response, L1_response]

# logL = make_heterodyne_likelihood_mutliple_detector(data_list, psd_list, response_list, gen_lensed_IMRPhenomD_polar, ref_param, H1_frequency, gmst, epoch, f_ref, 301)


n_dim = 15
n_chains = 1000
n_loop_training = 20
n_loop_production = 20
n_local_steps = 200
n_global_steps = 200
learning_rate = 0.001
max_samples = 100000
momentum = 0.9
num_epochs = 60
batch_size = 50000

# guess_param = ref_param

# guess_param = jnp.array(jnp.repeat(guess_param[None,:],int(n_chains),axis=0)*np.random.normal(loc=1,scale=0.1,size=(int(n_chains),n_dim)))
# guess_param[guess_param[:,1]>0.25,1] = 0.249
# guess_param[:,6] = (guess_param[:,6]%(2*jnp.pi))
# guess_param[:,7] = (guess_param[:,7]%(jnp.pi))
# guess_param[:,8] = (guess_param[:,8]%(jnp.pi))
# guess_param[:,9] = (guess_param[:,9]%(2*jnp.pi))


print("Preparing RNG keys")
rng_key_set = initialize_rng_keys(n_chains, seed=42)

print("Initializing MCMC model and normalizing flow model.")

prior_range = jnp.array([[20,40],[0.125,1.0],[-1,1],[-1,1],[0,5000],[-0.1,0.1],[0,2*np.pi],[-1,1],[0,np.pi],[0,2*np.pi],[-1,1], [0,5000], [5e-4,1], [0,1.49999],[0,1.49999]])
# optimize_prior_range = jnp.array([[20,40],[0.2,0.25],[-1,1],[-1,1],[0,2000],[-0.1,0.1],[0,2*np.pi],[0,np.pi],[0,np.pi],[0,2*np.pi],[-np.pi/2,np.pi/2]])


initial_position = jax.random.uniform(rng_key_set[0], shape=(int(n_chains), n_dim)) * 1
for i in range(n_dim):
    initial_position = initial_position.at[:,i].set(initial_position[:,i]*(prior_range[i,1]-prior_range[i,0])+prior_range[i,0])

# from ripple import Mc_eta_to_ms
# m1,m2 = jax.vmap(Mc_eta_to_ms)(guess_param[:,:2])
# q = m2/m1

# initial_position = initial_position.at[:,0].set(guess_param[:,0])

from astropy.cosmology import Planck18 as cosmo

z = np.linspace(0.002,3,10000)
dL = cosmo.luminosity_distance(z).value
dVdz = cosmo.differential_comoving_volume(z).value

def top_hat(x):
    output = 0.
    for i in range(n_dim):
        output = jax.lax.cond(x[i]>=prior_range[i,0], lambda: output, lambda: -jnp.inf)
        output = jax.lax.cond(x[i]<=prior_range[i,1], lambda: output, lambda: -jnp.inf)
    return output+jnp.log(jnp.interp(x[4],dL,dVdz))+jnp.log(jnp.interp(x[11],dL,dVdz))

def posterior(theta):
    q = theta[1]
    iota = jnp.arccos(theta[7])
    dec = jnp.arcsin(theta[10])
    prior = top_hat(theta)
    theta = theta.at[1].set(q/(1+q)**2) # convert q to eta
    theta = theta.at[7].set(iota) # convert cos iota to iota
    theta = theta.at[10].set(dec) # convert cos dec to dec
    return LogLikelihood(theta) + prior

model = RQSpline(n_dim, 10, [128,128], 8)

print("Initializing sampler class")

posterior = posterior

mass_matrix = jnp.eye(n_dim)
mass_matrix = mass_matrix.at[1,1].set(1e-3)
mass_matrix = mass_matrix.at[5,5].set(1e-3)
mass_matrix = mass_matrix.at[13,13].set(1e-2)
mass_matrix = mass_matrix.at[14,14].set(1e-2)

local_sampler = MALA(posterior, True, {"step_size": mass_matrix*3e-3})
print("Running sampler")

nf_sampler = Sampler(
    n_dim,
    rng_key_set,
    local_sampler,
    posterior,
    model,
    n_loop_training=n_loop_training,
    n_loop_production = n_loop_production,
    n_local_steps=n_local_steps,
    n_global_steps=n_global_steps,
    n_chains=n_chains,
    n_epochs=num_epochs,
    learning_rate=learning_rate,
    momentum=momentum,
    batch_size=batch_size,
    use_global=True,
    keep_quantile=0.,
    train_thinning = 40
)

nf_sampler.sample(initial_position)
chains, log_prob, local_accs, global_accs = nf_sampler.get_sampler_state().values()
np.savez('./result/GW150914_no_hetero.npz', chains=chains, log_prob=log_prob, local_accs=local_accs, global_accs=global_accs)

print("Local acceptance rate: ", np.mean(local_accs))
print("Global acceptance rate: ", np.mean(global_accs))

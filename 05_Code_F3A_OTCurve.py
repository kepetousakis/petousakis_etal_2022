import numpy as np
import Code_General_utility_spikes_pickling as util
import matplotlib.pyplot as plt
import Code_General_Nassi_functions as nf

_CND = 600
_ANF = 20  # +0% weight
_BNF = 20  # +0% weight
_DISP = 0
_N_NEURONS = 10
_N_RUNS = 10
_N_STIMULI = 18
_T_STIM = 2 # s
_T_STIM_PRESENT = 500 # ms
_DT = 0.1 # ms
_CONDITIONS = {200:"20:80",300:"30:70",400:"40:60",500:"50:50", 600:"60:40",700:"70:30",800:"80:20"}

_TRANSFORM_VECTOR = np.array([9,8,7,6,5,4,3,2,1,0,17,16,15,14,13,12,11,10,9])
_TRANSFORM_VECTOR = np.flip(_TRANSFORM_VECTOR)

neurons = [x for x in range(0,_N_NEURONS)]
runs = [y for y in range(0,_N_RUNS)]
stims = [z*10 for z in range(0,_N_STIMULI)]



firing_rates_all_neurons_all_runs = np.zeros(shape = (_N_NEURONS,_N_RUNS,_N_STIMULI))
firing_rates_all_neurons_across_runs = np.zeros(shape = (_N_NEURONS,_N_STIMULI))
firing_rates_across_neurons_across_runs = np.zeros(shape = (_N_STIMULI))

stds_all_neurons_all_runs = np.zeros(shape = (_N_NEURONS,_N_RUNS,_N_STIMULI))
stds_all_neurons_across_runs = np.zeros(shape = (_N_NEURONS,_N_STIMULI))
stds_across_neurons_across_runs = np.zeros(shape = (_N_STIMULI))

n_rates = []
n_errors = []
a_rates = []
a_errors = []

try:
	(n_rates,n_errors,a_rates,a_errors,firing_rates_all_neurons_all_runs) = util.pickle_load('./Data_F3A_OTCurve.pickle')
except:
	raise Exception
finally:
	global_rejections = 0
	for idx_n, nrn in enumerate(neurons):
		print(f'Neuron {nrn}', end = '  |  ')
		relevant_rates = np.squeeze(n_rates[idx_n,:])
		(nrn_pref, nrn_OSI, nrn_width, _ , _ ) = nf.tuning_properties(relevant_rates, [x*10 for x in range(0,18)])
		if nrn_OSI < 0.2 or nrn_width > 80 or np.isnan(nrn_OSI):
			verdict = 'REJECT'
			global_rejections += 1
		else:
			verdict = 'ACCEPT'
		print(f'Pref {nrn_pref} OSI {nrn_OSI:.3f} width {nrn_width} \t{verdict}')
		if verdict == 'REJECT':
			print('\t\t\t', end='')
			for x in relevant_rates:
				print(f'{x:.04}', end=', ')
			print()
	if global_rejections > 2:
		print(f'<!> Rejecting all neurons ({global_rejections}/10 neurons rejected).')
	else:
		print('<!> Neuron shows normal orientation tuning overall.')

# Figure 3A
fig = plt.figure()

if len(a_rates) == 0 or len(a_errors) == 0:
	a_rates = firing_rates_all_neurons_all_runs[:]
	a_errors = stds_all_neurons_all_runs[:]
else:
	a_rates = a_rates[:]
	a_errors = a_errors[:]

# x_axis = np.array([x for x in stims])
x_axis = np.array([x*10 for x in range(-9,10)])
y_axis = np.array([x for x in a_rates])
y_errors = np.array([x/np.sqrt(10) for x in a_errors])

preference = np.argmax(y_axis)*10

(nrn_pref, nrn_OSI, nrn_width, _ , _ ) = nf.tuning_properties(y_axis, [x*10 for x in range(0,18)])

print(f'OSI/width analysis: preferred {nrn_pref} deg, OSI {nrn_OSI}, width {nrn_width}')

print(f'Mean preferred orientation for disparity {_DISP}, condition {_CONDITIONS[_CND]}: {preference}')

# print(np.shape(x_axis), np.shape(y_axis), np.shape(y_errors))

plt.errorbar(x_axis, y_axis[_TRANSFORM_VECTOR], y_errors[_TRANSFORM_VECTOR], capsize=8, c='k')
plt.xticks(x_axis)
plt.xlabel('Stimulus orientation (deg)')
plt.ylabel('Neuronal response (Hz)')
plt.title('Figure 3A')
fig.set_tight_layout(True)
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()

plt.show()


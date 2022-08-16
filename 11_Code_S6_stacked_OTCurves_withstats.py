#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 15:40:19 2022

@author: kostasp
"""

import matplotlib.pyplot as plt
import numpy as np
import Code_General_utility_spikes_pickling as util
import Code_General_Nassi_functions as nf
import scipy.stats as stats
import matplotlib
matplotlib.rcParams.update({'errorbar.capsize': 2})

def eval_tuning(firing_rates_all_neurons_across_runs):
	adjusted_rates = firing_rates_all_neurons_across_runs
	prefs = []
	OSIs = []
	widths = []
	verdicts= []
	neurons = [x for x in range(0,np.min(np.shape(firing_rates_all_neurons_across_runs)))]
	global_rejections = 0
	for idx_n, nrn in enumerate(neurons):
		print(f'Neuron {nrn}', end = '  |  ')
		relevant_rates = np.squeeze(firing_rates_all_neurons_across_runs[idx_n,:])
		(nrn_pref, nrn_OSI, nrn_width, _ , _ ) = nf.tuning_properties(relevant_rates, [x*10 for x in range(0,18)])
		prefs.append(nrn_pref); OSIs.append(nrn_OSI); widths.append(nrn_width)
		if nrn_OSI < 0.2 or nrn_width > 80 or np.isnan(nrn_OSI):
			verdict = 'REJECT'
			global_rejections += 1
			adjusted_rates[idx_n,:] = np.nan
		else:
			verdict = 'ACCEPT'
		verdicts.append(verdict)
		print(f'Pref {nrn_pref} OSI {nrn_OSI:.3f} width {nrn_width} \t{verdict}')
		if verdict == 'REJECT':
			print('\t\t\t', end='')
			for x in relevant_rates:
				print(f'{x:.04}', end=', ')
			print()
	if global_rejections > 2:
		print(f'<!> Rejecting all neurons ({global_rejections}/10 neurons rejected).')
# 		adjusted_rates = np.empty(shape=np.shape(firing_rates_all_neurons_across_runs))
# 		adjusted_rates[:] = np.nan
	else:
		print('<!> Neuron shows normal orientation tuning overall.')
		
	return adjusted_rates, prefs, OSIs, widths, verdicts

def test_stats_add_stars(labels,data,feature,starheight,staroffset,offsetmove, comparewith=-1):
	# Statistical testing with paired two-sample t-test
	N = len(data)
	
	xx = [x for x in range(0, len(labels))]
	xraw = [x for x in range(0, len(labels))]
	
	results = {x:{y:0 for y in labels} for x in labels}  # table-style dict for p-values - comparisons are in the style of X to Y, accessing the result via results[x][y]
	res_array = np.zeros(shape = (N,N))
	res_array_bin = np.zeros(shape = (N,N))
	test = stats.ttest_ind
	verdict = lambda pval: '*' if pval <= 0.05 else 'NS'
	if comparewith == -1:
		comparator_labels = labels
	else:
		if type([]) != type(comparewith):
			comparewith = [comparewith]
		comparator_labels = [labels[x] for x in comparewith]
	for i,comparator in enumerate(comparator_labels):
		for j,comparand in enumerate(labels):
			if comparator == comparand:
				res_array[i][j] = -1
				res_array_bin[i][j] = -1
			else:
				(stat, pval) = test(data[i][feature], data[j][feature])
				print(f'Testing {comparator} against {comparand}. Results: Statistic {stat:2.4f}, p-value {pval:2.8f} | Verdict: \t {verdict(pval)}')
				results[comparator][comparand] = pval
				res_array[i][j] = pval
				if not _BONFERRONI:
					if pval <= 0.05:
						res_array_bin[i][j] = 1
					if pval <= 0.01:
						res_array_bin[i][j] = 2
					if pval <= 0.001:
						res_array_bin[i][j] = 3
					if pval > 0.05:
						res_array_bin[i][j] = 0
				else:
					alpha_bf = 0.05/(8)  # control compared with the rest is 8 comparisons
					if pval <= alpha_bf:
						res_array_bin[i][j] = 1
					else:
						res_array_bin[i][j] = 0
				print(f'Testing {comparator} against {comparand}. Results: Statistic {stat:2.4f}, p-value {pval:2.4f} | Verdict: \t {"*" if len(int(res_array_bin[i][j])*"*")>0 else "NS"}')
				
	idxtolabel = {x:y for x,y in zip(xraw, labels)}
	idxtopos = {x:y for x,y in zip(xraw, xx)}
	
	results_triu = np.triu(res_array_bin)
	lineardata = []
	for datum in [x[feature] for x in data]:
		lineardata += [x for x in datum]
	starting_height = max(lineardata) + starheight
	offset = staroffset
	
	for i in range(0,N):
		for j in range(0,N):
			if results_triu[i,j] > 0:
				offset += offsetmove
				print(f'>0 for i={i} and j={j} ( comparator={labels[i]} and comparand={labels[j]} )')
				height_diff = starting_height - max(data[i][feature]+data[j][feature]) -(offset)
				xvals = [idxtopos[i], idxtopos[j]]
				yvals = [starting_height-height_diff, starting_height-height_diff]
				plt.plot(xvals, yvals, 'k')
				print(f"Placing star at {np.mean(xvals)} and {yvals[0]}+")
				plt.text(np.mean(xvals), yvals[0], '*'*int(res_array_bin[i][j]), fontsize='xx-large', ha='center')  # ha = horizontal alignment
	return results
				

_TRANSFORM_VECTOR = np.array([9,8,7,6,5,4,3,2,1,0,17,16,15,14,13,12,11,10,9])
_TRANSFORM_VECTOR = np.flip(_TRANSFORM_VECTOR)

_BONFERRONI = False

file_root = './Figure_5_SuppFigure_6_Data/'

file_Bnull = f'{file_root}attx0_bttx1_nsA0_nsB0/firing_rates_stds_cnd600.pickle'
file_Anull = f'{file_root}attx1_bttx0_nsA0_nsB0/firing_rates_stds_cnd600.pickle'
file_nsB = f'{file_root}attx0_bttx0_nsA0_nsB1/firing_rates_stds_cnd600.pickle'
file_nsA = f'{file_root}attx0_bttx0_nsA1_nsB0/firing_rates_stds_cnd600.pickle'
file_control = f'{file_root}attx0_bttx0_nsA0_nsB0/firing_rates_stds_cnd600.pickle'
file_spikeblockade = f'{file_root}spike_removals_variablenoise/cnd600_firingrates.pickle'
file_ampanmdaA = f'{file_root}ampaAsf10_nmdaAsf10_ampaBsf100_nmdaBsf100/firing_rates_stds_cnd600.pickle'
file_ampanmdaB = f'{file_root}ampaAsf100_nmdaAsf100_ampaBsf10_nmdaBsf10/firing_rates_stds_cnd600.pickle'
file_syninterventions = f'{file_root}apical_basal_ampa-nmda_interventions_fct05/cnd600_firingrates.pickle'


(ctrl_n_rates,ctrl_n_errors,ctrl_a_rates,ctrl_a_errors,ctrl_firing_rates_all_neurons_all_runs) = util.pickle_load(file_control)
(Anull_n_rates,Anull_n_errors,Anull_a_rates,Anull_a_errors,Anull_firing_rates_all_neurons_all_runs) = util.pickle_load(file_Anull)
(Bnull_n_rates,Bnull_n_errors,Bnull_a_rates,Bnull_a_errors,Bnull_firing_rates_all_neurons_all_runs) = util.pickle_load(file_Bnull)
(nsA_n_rates,nsA_n_errors,nsA_a_rates,nsA_a_errors,nsA_firing_rates_all_neurons_all_runs) = util.pickle_load(file_nsA)
(nsB_n_rates,nsB_n_errors,nsB_a_rates,nsB_a_errors,nsB_firing_rates_all_neurons_all_runs) = util.pickle_load(file_nsB)
spikeblockade_firing_rates_all_neurons_all_runs = util.pickle_load(file_spikeblockade)
(ampanmdaA_n_rates,ampanmdaA_n_errors,ampanmdaA_a_rates,ampanmdaA_a_errors,ampanmdaA_firing_rates_all_neurons_all_runs) = util.pickle_load(file_ampanmdaA)
(ampanmdaB_n_rates,ampanmdaB_n_errors,ampanmdaB_a_rates,ampanmdaB_a_errors,ampanmdaB_firing_rates_all_neurons_all_runs) = util.pickle_load(file_ampanmdaB)
syninterventions_firing_rates_all_neurons_all_runs = util.pickle_load(file_syninterventions)

# labels = ['Control','Apical gNa nullified','Basal gNa nullified','Apical stim-driven synapses nullfied','Basal stim-driven synapses nullfied', 'Spikes surviving Anull', 'Spikes surviving Bnull', 
# 		  'Apical stim-driven AMPA/NMDA 90% block','Basal stim-driven AMPA/NMDA 90% block']
# labels = ['Control','attx','bttx','nsA','nsB', 'Anull (intv)', 'Bnull (intv)', 
#		  'ampaAsf/nmdaAsf 0.1','ampaBsf/nmdaBsf 0.1', 'SynIntvA 0.5', 'SynIntvB 0.5']
labels = ['Control', 'Apical, no stim-driven', 'Basal, no stim-driven', 'Apical, 90% reduction', 'Basal, 90% reduction','Apical, Sodium intervention', 'Basal, Sodium intervention',
		  'Apical, sodium block', 'Basal, sodium block', 'Apical, 50% intervention', 'Basal, 50% intervention']
all_rates = [ctrl_firing_rates_all_neurons_all_runs,Anull_firing_rates_all_neurons_all_runs,Bnull_firing_rates_all_neurons_all_runs,nsA_firing_rates_all_neurons_all_runs,nsB_firing_rates_all_neurons_all_runs,
			 spikeblockade_firing_rates_all_neurons_all_runs[0],spikeblockade_firing_rates_all_neurons_all_runs[1],ampanmdaA_firing_rates_all_neurons_all_runs,ampanmdaB_firing_rates_all_neurons_all_runs,
			 syninterventions_firing_rates_all_neurons_all_runs[0], syninterventions_firing_rates_all_neurons_all_runs[1]] #spikeblockade is Anull + Bnull

case_rates = []
case_errors = []
case_properties = []


for case, label in enumerate(labels):
	nrn_rates_raw = np.mean(all_rates[case],axis=1)
	adjusted_rates, prefs, OSIs, widths, verdicts = eval_tuning(nrn_rates_raw)
	tuned = sum(['REJECT' in x for x in verdicts]) <= 2
	N = sum(['ACCEPT' in x for x in verdicts])
	nrn_rates_filtered = np.nanmean(adjusted_rates,axis=0)
	case_rates.append(nrn_rates_filtered)
	case_errors.append(np.nanstd(adjusted_rates,axis=0))
	case_properties.append([label, prefs, OSIs, widths, verdicts, tuned, N])
	print(f'Case "{label}" is {"not " if not tuned else ""}tuned.')


# Supp.Figure 6A
fig = plt.figure()
fig.set_tight_layout(True)
plt.title('Supp.Figure 6A (differently colored)')				

for i, (rates, errors) in enumerate(zip(case_rates, case_errors)):
		
	x_axis = np.array([x*10 for x in range(-9,10)])
	y_axis = np.array([x for x in rates])
	y_errors = np.array([x/np.sqrt(case_properties[i][-1]) for x in errors])
	
	preference = np.argmax(y_axis)*10
	
	
	print(f'Mean preferred orientation: {preference}')
	
	plt.errorbar(x_axis, y_axis[_TRANSFORM_VECTOR], y_errors[_TRANSFORM_VECTOR], label=labels[i])
	plt.xticks(x_axis)
	plt.xlabel('Stimulus orientation (deg)')
	plt.ylabel('Neuronal response (Hz)')
	plt.legend()

figmanager = plt.get_current_fig_manager()
figmanager.window.showMaximized()

	
mean_OSIs = [np.mean(case_properties[x][2]) for x in range(0,len(labels))]
std_OSIs = [np.std(case_properties[x][2])/np.sqrt(case_properties[x][6]) for x in range(0,len(labels))]
mean_widths = [np.mean(case_properties[x][3]) for x in range(0,len(labels))] # If the mean OSI is nan, the width doesn't matter.
std_widths = [np.std(case_properties[x][3])/np.sqrt(case_properties[x][6]) for x in range(0,len(labels))]


for i,x in enumerate(case_properties):
	if not x[-2]:
		mean_OSIs[i] = np.nan
		mean_widths[i] = np.nan
		std_OSIs[i] = np.nan
		std_widths[i] = np.nan
		case_properties[i][2] = [np.nan for x in case_properties[i][3]]
		case_properties[i][3] = [np.nan for x in case_properties[i][3]]

# Supp.Figur 6B
fig = plt.figure()
fig.set_tight_layout(True)
plt.title('Supp.Figure 6B (uncolored)')
plt.bar([x for x in range(0,len(labels))], mean_OSIs, width=0.2, align='center', yerr=std_OSIs)
res = test_stats_add_stars(labels,case_properties,2,0.07,0,0.05,comparewith=0)
plt.xticks([x for x in range(0,len(labels))], labels, fontsize=6, rotation=45)
plt.ylabel('Mean OSI value')
plt.ylim([0,1.2])
figmanager = plt.get_current_fig_manager()
figmanager.window.showMaximized()

plt.show()

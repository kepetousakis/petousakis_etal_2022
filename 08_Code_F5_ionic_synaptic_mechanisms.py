# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 13:24:28 2021

@author: KEPetousakis
"""

import Code_General_utility_spikes_pickling as util
import numpy as np
import Code_General_Nassi_functions as nf
import matplotlib.pyplot as plt
from copy import deepcopy as dcp
import scipy.stats as stats

_BONFERRONI = False

def test_stats_add_stars_generic(labels,data,feature,starheight,staroffset,offsetmove, comparewith=-1):
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
# 				(stat, pval) = test(data[i][feature], data[j][feature])
				(stat, pval) = test(data[i], data[j])
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
					alpha_bf = 0.05/(8)  # control compared with the rest is 8 comparisons, regardless of what is displayed in each subplot
					if pval <= alpha_bf:
						res_array_bin[i][j] = 1
					else:
						res_array_bin[i][j] = 0
				print(f'Testing {comparator} against {comparand}. Results: Statistic {stat:2.4f}, p-value {pval:2.4f} | Verdict: \t {"*" if len(int(res_array_bin[i][j])*"*")>0 else "NS"}')
				
	idxtolabel = {x:y for x,y in zip(xraw, labels)}
	idxtopos = {x:y for x,y in zip(xraw, xx)}
	
	results_triu = np.triu(res_array_bin)
	lineardata = []
	for datum in [x for x in data]:
		lineardata += [x for x in datum]
	starting_height = 1
# 	starting_height = max(lineardata) + starheight
	offset = 0.05
	
	for i in range(0,N):
		for j in range(0,N):
			if results_triu[i,j] > 0:
				offset += offsetmove
				print(f'>0 for i={i} and j={j} ( comparator={labels[i]} and comparand={labels[j]} )')
# 				height_diff = starting_height - max(data[i]+data[j]) -(offset)
				height_diff = offset
				xvals = [idxtopos[i], idxtopos[j]]
				yvals = [starting_height-height_diff, starting_height-height_diff]
				plt.plot(xvals, yvals, 'k')
				print(f"Placing star at {np.mean(xvals)} and {yvals[0]}+")
				plt.text(np.mean(xvals), yvals[0], '*'*int(res_array_bin[i][j]), fontsize='xx-large', ha='center')  # ha = horizontal alignment
	return results

def lookup(target_dict, search_parameter):
	
	keys = target_dict.keys()
	matches = {}
	for key in keys:
		if search_parameter in key:
			matches[key] = target_dict[key]
	return matches

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
	else:
		print('<!> Neuron shows normal orientation tuning overall.')
		
	return adjusted_rates, prefs, OSIs, widths, verdicts

dt = 0.1
runtime = 2500
onset = 500

CND = 600
NRNS = [x for x in range(0,10)]
RUNS = [x for x in range(0,10)]
STIMS = [x*10 for x in range(0,18)]
CASES = ['Anull','Bnull']
CASES_LABELS = ['Basal target', 'Apical target']

_CONDITIONS = {100:"10b:90a",200:"20b:80a",300:"30b:70a",400:"40b:60a",500:"50b:50a", 600:"60b:40a",700:"70b:30a",800:"80b:20a",900:"90b:10a"}

outcome_dict = {0:'apical',1:'basal',2:'unstable',3:'bistable'}

_TRANSFORM_VECTOR = np.array([9,8,7,6,5,4,3,2,1,0,17,16,15,14,13,12,11,10,9])
_TRANSFORM_VECTOR = np.flip(_TRANSFORM_VECTOR)

filepath_root = './Figure_5_SuppFigure_6_Data'

# Control curve data
file_control = f'{filepath_root}/attx0_bttx0_nsA0_nsB0/firing_rates_stds_cnd600.pickle'
(ctrl_n_rates,ctrl_n_errors,ctrl_a_rates,ctrl_a_errors,ctrl_firing_rates_all_neurons_all_runs) = util.pickle_load(file_control)
x_axis_ctrl = np.array([x*10 for x in range(-9,10)])
y_axis_ctrl = np.array([x for x in ctrl_a_rates])
y_errors_ctrl = np.array([x/np.sqrt(10) for x in ctrl_a_errors])
nrn_rates_raw = np.mean(ctrl_firing_rates_all_neurons_all_runs,axis=1)
ctrl_adjusted_rates, ctrl_prefs, ctrl_OSIs, ctrl_widths, ctrl_verdicts = eval_tuning(nrn_rates_raw)
tuned_ctrl = sum(['REJECT' in x for x in ctrl_verdicts]) <= 2
N_ctrl = sum(['ACCEPT' in x for x in ctrl_verdicts])

mean_OSIs_ctrl = np.mean(ctrl_OSIs)
std_OSIs_ctrl = np.std(ctrl_OSIs)/np.sqrt(N_ctrl)

# No stim-driven synapses (Figure 5A)
file_root = filepath_root
file_nsB = f'{file_root}/attx0_bttx0_nsA0_nsB1/firing_rates_stds_cnd600.pickle'
file_nsA = f'{file_root}/attx0_bttx0_nsA1_nsB0/firing_rates_stds_cnd600.pickle'
(nsA_n_rates,nsA_n_errors,nsA_a_rates,nsA_a_errors,nsA_firing_rates_all_neurons_all_runs) = util.pickle_load(file_nsA)
(nsB_n_rates,nsB_n_errors,nsB_a_rates,nsB_a_errors,nsB_firing_rates_all_neurons_all_runs) = util.pickle_load(file_nsB)

labels = ['Control', 'Apical target', 'Basal target']
all_rates = [ctrl_firing_rates_all_neurons_all_runs,nsA_firing_rates_all_neurons_all_runs,nsB_firing_rates_all_neurons_all_runs]

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

fig, ax = plt.subplots(1,2, gridspec_kw={'width_ratios':[2,1]})
fig.set_tight_layout(True)
plt.suptitle('Figure 5A (minor color/cosmetic differences)')	
plt.sca(ax[0])			

for i, (rates, errors) in enumerate(zip(case_rates, case_errors)):
		
	x_axis = np.array([x*10 for x in range(-9,10)])
	y_axis = np.array([x for x in rates])
	y_errors = np.array([x/np.sqrt(case_properties[i][-1]) for x in errors])
	
	preference = np.argmax(y_axis)*10
	
	
	print(f'Mean preferred orientation: {preference}')
	
	plt.errorbar(x_axis, y_axis[_TRANSFORM_VECTOR], y_errors[_TRANSFORM_VECTOR], label=labels[i], capsize=2)
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
plt.sca(ax[1])
plt.bar([x for x in range(0,len(labels))], mean_OSIs, width=0.2, align='center', yerr=std_OSIs, capsize=2)
res = test_stats_add_stars_generic(labels,case_properties,2,0.07,0,0.05,comparewith=0)
plt.xticks([x for x in range(0,len(labels))], labels)
plt.ylabel('Mean OSI value')
plt.ylim([0, 1.1])
figmanager = plt.get_current_fig_manager()
figmanager.window.showMaximized()

#%%

filepath_verdicts = []
filepath_timings = []
suptitles = ['Figure 5B (minor color/cosmetic differences)', 'Figure 5C (minor color/cosmetic differences)']

# Stim-driven synapse weight reduction by 50% (intervention) (Figure 5B)
filepath_verdicts.append(f'{filepath_root}/apical_basal_ampa-nmda_interventions_fct05/spike_survival_verdicts_stimintv_sttx0.pickle')
filepath_timings.append(f'{filepath_root}/apical_basal_ampa-nmda_interventions_fct05/pre_intervention_spike_timings_stimintv_sttx0.pickle')
# Nullification of sodium conductance (intervention) (Figure 5C)
filepath_verdicts.append(f'{filepath_root}/spike_removals_variablenoise/spike_survival_verdicts_allstims_sttx1.pickle')
filepath_timings.append(f'{filepath_root}/spike_removals_variablenoise/pre_intervention_spike_timings_allstims_sttx1.pickle')

for idx_f, (filepath, filepath2) in enumerate(zip(filepath_verdicts, filepath_timings)):
	
	verdicts = util.pickle_load(filepath)
	timings = util.pickle_load(filepath2)
	
	timings_copy = dcp(timings)
	verdicts_copy = dcp(verdicts)
	
	for entry in timings_copy.keys():
		if timings[entry][1] < onset/dt:
			del timings[entry]
			del verdicts[entry]
			
	firing_rates = np.zeros(shape=(2,10,10,18)) # nullification(0=Anull,1=Bnull), neuron, run, stim
	# firing_rates = np.zeros(shape=(10,10,18)) # neuron, run, stim
	
	OSIs_all = np.zeros(shape=(2,10))
	widths_all = np.zeros(shape=(2,10))
	
	fig, ax = plt.subplots(1,2, gridspec_kw={'width_ratios':[2,1]})
	plt.suptitle(suptitles[idx_f])
	fig.set_tight_layout(True)
	
	ax[0].errorbar(x_axis_ctrl, y_axis_ctrl[_TRANSFORM_VECTOR], y_errors_ctrl[_TRANSFORM_VECTOR], label='Control', capsize=2)
	plt.sca(ax[0])
	plt.xticks(x_axis_ctrl)
	
	for iC, case in enumerate(CASES):
		for iN, nrn in enumerate(NRNS):
			for iR, run in enumerate(RUNS):
				for iS, stim in enumerate(STIMS):
					key = f'{CND}-{nrn}-{run}-{stim}-'
					matches = lookup(verdicts,key)
					case_spikes = 0
					for match_key in matches.keys():
						if matches[match_key][case] == 1:
							case_spikes +=1
					firing_rates[iC,iN,iR,iS] = case_spikes/2
		
		mean_firing_across_runs = np.nanmean(firing_rates[iC], axis=1)
		adjusted_rates, prefs, OSIs, widths, labels = eval_tuning(mean_firing_across_runs)
		OSIs_all[iC,:] = np.array(OSIs)
		widths_all[iC,:] = np.array(widths)
		N = sum([x=='ACCEPT' for x in labels])
		if N < 8: # filter out OSIs of untuned neurons
			OSIs_all[iC,:] = np.array([np.nan for x in range(0,len(OSIs))])
		
		mean_firing_across_neurons = np.nanmean(mean_firing_across_runs, axis=0)
		std_firing_across_neurons = np.nanstd(mean_firing_across_runs, axis=0)
		
		x_axis = np.array([x*10 for x in range(-9,10)])
		y_axis = np.array([x for x in mean_firing_across_neurons])
		y_errors = np.array([x/np.sqrt(N) for x in std_firing_across_neurons])
		
		preference = np.argmax(y_axis)*10
		
		print(f'Mean preferred orientation: {preference}')
		
		ax[0].errorbar(x_axis, y_axis[_TRANSFORM_VECTOR], y_errors[_TRANSFORM_VECTOR], label=CASES_LABELS[iC], capsize=2)
		plt.sca(ax[0])
		plt.xticks(x_axis)
	
	plt.sca(ax[0])	
	plt.xlabel('Stimulus orientation (deg)')
	plt.ylabel('Neuronal response (Hz)')
		
	plt.legend()
	figmanager = plt.get_current_fig_manager()
	figmanager.window.showMaximized()


	_CASES = ['Ctrl','Anull','Bnull']
	_CASES_LABELS = ['Control','Basal target', 'Apical target']
	
	OSIs_all = np.concatenate((np.reshape(ctrl_OSIs, newshape=(1,10)), OSIs_all), axis=0)

	ax[1].bar([x for x in range(0,len(_CASES))], np.mean(OSIs_all, axis=1), width=0.2, align='center', yerr=np.std(OSIs_all, axis=1)/np.sqrt(10), capsize=2)
	plt.sca(ax[1])
	res = test_stats_add_stars(_CASES_LABELS,OSIs_all,0,0.07,0,0.05)
	plt.xticks([x for x in range(0,len(_CASES_LABELS))],_CASES_LABELS)
	plt.ylabel('Mean OSI value')
	plt.ylim([0, 1.1])

	
plt.show()
	


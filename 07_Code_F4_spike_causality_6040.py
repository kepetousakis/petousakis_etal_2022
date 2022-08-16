# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 13:24:28 2021

@author: KEPetousakis
"""

import Code_General_utility_spikes_pickling as util
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy as dcp

CNDS = [600]
testcnd = 600
BNFS = {x:[20] for x in CNDS}
NRNS = [x for x in range(0,50)]
BNFS_PLOT = {x:[0] for x in CNDS}
_CONDITIONS = {100:"10b:90a",200:"20b:80a",300:"30b:70a",400:"40b:60a",500:"50b:50a", 600:"60b:40a",700:"70b:30a",800:"80b:20a",900:"90b:10a"}

outcome_dict = {0:'apical',1:'basal',2:'unstable',3:'bistable'}


def lookup(target_dict, search_parameter):
	
	keys = target_dict.keys()
	matches = {}
	for key in keys:
		if search_parameter in key:
			matches[key] = target_dict[key]
	return matches

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

_FIGURES = ['Figure 4B', 'Figure 4D', 'Figure 4C', 'Figure 4E']
filepath_root = './Figure_4_SuppFigure_5_Data'
filepath_control_pref = f'{filepath_root}/dt01_interventions_variablenoise/spike_survival_verdicts_syndistr_sttx1.pickle'
filepath_control_orth = f'{filepath_root}/dt01_interventions_variablenoise_stim90/spike_survival_verdicts_syndistr_sttx1_stim90.pickle'
filepath_equalnoise_pref = f'{filepath_root}/dt01_interventions_fixednoise/spike_survival_verdicts_syndistr_fixednoise_sttx1.pickle'
filepath_equalnoise_orth = f'{filepath_root}/dt01_interventions_fixednoise_stim90/spike_survival_verdicts_syndistr_fixednoise_sttx1_stim90.pickle'

angle = -1

verdicts_all = [util.pickle_load(filepath_control_pref),util.pickle_load(filepath_control_orth),util.pickle_load(filepath_equalnoise_pref), util.pickle_load(filepath_equalnoise_orth)]

for idx_v, verdicts in enumerate(verdicts_all):

	results = np.zeros( shape = (len(CNDS), len(BNFS[testcnd]), len(NRNS), 5) )  # 5 = 0(apical), 1(basal), 2(unstable), 3(bistable) - each holds a count (number of occurrences) - 5 is the sum
	summation = np.zeros( shape = (len(CNDS), len(BNFS[testcnd]) ) )
	fractions = np.zeros( shape = (len(CNDS), len(BNFS[testcnd]), 4) )
	fractions_per_neuron = np.zeros( shape = (len(CNDS), len(BNFS[testcnd]), len(NRNS), 4) )
	cross_nrn_averages = np.zeros( shape = (len(CNDS), len(BNFS[testcnd]), 4) )
	cross_nrn_stds = np.zeros( shape = (len(CNDS), len(BNFS[testcnd]), 4) )
	total_fractions = np.zeros( shape = (len(CNDS), 4))
	
	total_spikes = 0
	cnd_sums = []
	
	for ic, cnd in enumerate(CNDS):
		for ib, bnf in enumerate(BNFS[cnd]):
			cnd_sum = 0
			for nrn in NRNS:
				if angle < 0:
					search_parameter = f'{cnd}-{bnf}-{nrn}-'
				else:
					search_parameter = f'{cnd}-{bnf}-{nrn}-{angle}-'
				matches = lookup(verdicts, search_parameter)
				matches = [x for x in matches.values()]
				matches = [x['Verdict'] for x in matches]
				apicals   = sum(['apical' in x for x in matches])
				basals    = sum(['basal' in x for x in matches])
				unstables = sum(['unstable' in x for x in matches])
				bistables = sum(['bistable' in x for x in matches])
				results[ic][ib][nrn][0] = apicals
				results[ic][ib][nrn][1] = basals
				results[ic][ib][nrn][2] = unstables
				results[ic][ib][nrn][3] = bistables
				sum_res = apicals+basals+unstables+bistables
				results[ic][ib][nrn][4] = sum_res
				partialsum = apicals + basals + unstables + bistables
				cnd_sum += partialsum
				total_spikes += partialsum
				print(f'CND {cnd}, BNF {bnf}, NRN {nrn}: Apical: {apicals}\t Basal: {basals}\t Unstable: {unstables}\t Bistable: {bistables}')
				
				fractions[ic][ib][0] += apicals
				fractions[ic][ib][1] += basals
				fractions[ic][ib][2] += unstables
				fractions[ic][ib][3] += bistables
				summation[ic][ib]    += sum_res
				
				if sum_res != 0:
					fractions_per_neuron[ic][ib][nrn][0] = apicals/sum_res
					fractions_per_neuron[ic][ib][nrn][1] = basals/sum_res
					fractions_per_neuron[ic][ib][nrn][2] = unstables/sum_res
					fractions_per_neuron[ic][ib][nrn][3] = bistables/sum_res
				else:
					fractions_per_neuron[ic][ib][nrn][0] = 0
					fractions_per_neuron[ic][ib][nrn][1] = 0
					fractions_per_neuron[ic][ib][nrn][2] = 0
					fractions_per_neuron[ic][ib][nrn][3] = 0
			
			cnd_sums.append(cnd_sum)
			
			if summation[ic][ib] != 0:
				fractions[ic][ib][0] /= summation[ic][ib]
				fractions[ic][ib][1] /= summation[ic][ib]
				fractions[ic][ib][2] /= summation[ic][ib]
				fractions[ic][ib][3] /= summation[ic][ib]
			else:
				fractions[ic][ib][0] = 0
				fractions[ic][ib][1] = 0
				fractions[ic][ib][2] = 0
				fractions[ic][ib][3] = 0
				
		total_fractions[ic][0] = sum(sum(results[ic,:,:,0]))/sum(sum(results[ic,:,:,4]))
		total_fractions[ic][1] = sum(sum(results[ic,:,:,1]))/sum(sum(results[ic,:,:,4]))
		total_fractions[ic][2] = sum(sum(results[ic,:,:,2]))/sum(sum(results[ic,:,:,4]))
		total_fractions[ic][3] = sum(sum(results[ic,:,:,3]))/sum(sum(results[ic,:,:,4]))
		
	for ic,cnd in enumerate(CNDS):
		print(f'CND {cnd}: Apical: {total_fractions[ic][0]*100:.2f}%\t Basal: {total_fractions[ic][1]*100:.2f}%\t Unstable: {total_fractions[ic][2]*100:.2f}%\t Bistable: {total_fractions[ic][3]*100:.2f}%')
	print(f'Total spikes: {total_spikes}')

	
	# Average across neurons to get mean + std
	for ic,cnd in enumerate(CNDS):
		for ib,bnf in enumerate(BNFS[cnd]):
			for outcome in range(0,4):
				cross_nrn_averages[ic][ib][outcome] = np.mean(fractions_per_neuron[ic,ib,:,outcome])
				cross_nrn_stds[ic][ib][outcome] = np.std(fractions_per_neuron[ic,ib,:,outcome])/np.sqrt(50)


	
	# Figures 4B, 4C, 4D, 4E
	fig,ax = plt.subplots()
# 	labels = ['Apical','Basal','Unstable','Bistable']
	labels = ['Apical','Basal','Cooperative','']
# 	x_offset = [x/10 for x in range(-4,6)]
	for ic, cnd in enumerate(CNDS):
		this_res = total_fractions[ic,:]
	
		x = np.arange(len(labels))
		width = 0.1
# 		rects_this = ax.bar(x + x_offset[ic], this_res, width, label = f'{_CONDITIONS[cnd]}')
		rects_this = ax.bar(x, this_res, width, label = f'{_CONDITIONS[cnd]}', color='k')
		ax.set_xticks(x)
		ax.set_xticklabels(labels)
# 		ax.legend()
		
		autolabel(rects_this)
	
	plt.xlabel('Intervention verdict')
	plt.ylabel('Fraction of spikes')
	plt.title(f'{_FIGURES[idx_v]}')
	plt.ylim([0,1])
# 	plt.legend([_CONDITIONS[x] for x in CNDS])
	fig.set_tight_layout(True)
	figmanager = plt.get_current_fig_manager()
	figmanager.window.showMaximized()
	
plt.show()
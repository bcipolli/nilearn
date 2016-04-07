# *- encoding: utf-8 -*-
# Author: Ben Cipollini
# License: BSD
"""
How many ics? Find the sweet spot where the
average dissimilarity between unilateral ('RL') and
bilateral ('wb') components is minimized.

Do that with a hard loop on the # of components, then
plotting the mean dissimilarity (derived from the
score matrix).
"""

import os.path as op

import matplotlib.pyplot as plt
import numpy as np
import re

from main import main
from nibabel_ext import NiftiImageWithTerms
from nilearn_ext.masking import HemisphereMasker
from nilearn_ext.plotting import save_and_close
from sklearn.externals.joblib import Memory


def do_ic_loop(components, scoring, dataset, noSimScore=False,
               memory = Memory(cachedir='nilearn_cache'), **kwargs):
    score_mats = []
    images = []
    images_key = ["R","L","wb"]
    umis = []
    out_dir = op.join('ica_imgs', dataset)
    for c in components:
        if noSimScore: # only calculate sparsity and HPI using ICA images (i.e. do not run main.py)
            print("Simply loading component images for n_component = %s"% c)
            img_d = {}
            nii_dir = op.join('ica_nii', dataset, str(c))
            for k in images_key:
                img_path = op.join(nii_dir, '%s_ica_components.nii.gz'%(k))
                img_d[k] = NiftiImageWithTerms.from_filename(img_path)
            images.append(img_d)
        else:
            print("Running analysis with %d components." % c)
            img_d, score_mats_d, umi_d = main(n_components=c, dataset=dataset,
                                        scoring=scoring, **kwargs)
            score_mats.append(score_mats_d)
            images.append(img_d)
            umis.append(umi_d)

    # Now we have all the scores; now get values of interest and plot.
    
    if not noSimScore:
        # 1) mean dissimilarity scores for the best match pairs
        sim_scores = {key:[] for key in score_mats[0]}
        um_R, um_L = [], []             # Number of unmatched R and L components
        
        for sm_d, umi_d in zip(score_mats, umis):
        
            for key in sim_scores:
                sm = sm_d[key]
                sm[sm == 0] = np.inf  # is this necessary??
                sim_scores[key].append(sm.min(axis=1).mean())
                
            num_unmatched_R, num_unmatched_L = [len(umi_d[k]) for k in ['R','L']]
            um_R.append(num_unmatched_R)
            um_L.append(num_unmatched_L)
        
        fh = plt.figure(figsize=(10, 10))
        ax1 = fh.gca()
        ax1.plot(components, np.asarray(sim_scores.values()).T)
        ax1.legend(sim_scores.keys())
        ax1.set_xlabel("# of components"), ax1.set_ylabel("mean %s scores" % scoring)
        ax2 = ax1.twinx()
        ax2.plot(components, um_R, 'r--', label = 'unmatched R' )
        ax2.plot(components, um_L, 'b--', label = 'unmatched L' )
        ax2.set_ylabel("# of unmatched R- or L- components")
        ax2.set_ylim(ymax = (max(max(um_R),max(um_L))+9)//10*10)
        ax2.legend(loc = 4)
        plt.title("Average dissimilarity scores for the best-match pairs")
        
        out_path = op.join(out_dir, '%s_simscores.png' % scoring)
        save_and_close(out_path, fh=fh)
    
    
    # 2) overall sparsity of the wb components
    sparsity_levels = ['pos_005','neg_005','abs_005']
    
    # For each component type, store sparsity values for each sparsity level
    sparsity = {k:{s:[] for s in sparsity_levels} for k in images_key}  
    
    # 3) hemispheric participation index (HPI): (R-L)/(R+L) for # of voxels>0.005
    # calculated from wb components  
    hemi_maskers = [HemisphereMasker(hemisphere=hemi, memory=memory).fit() for hemi in ['R','L']]
    x = [[c]*c for c in components]
    y_vals = {sign:{val:[] for val in ['vals','mean','sd']} for sign in ['pos','neg']}  
    
    for img_d in images:    
        # 2) get mean sparsity per component
        for k in images_key:
            img = img_d[k]
            for s in sparsity_levels:
                thresh = float('0.%s'%(re.findall('\d+',s)[0]))
                # sparsity_vals is a list containing # of voxels above the given sparsity level for each component
                if 'pos' in s:
                    sparsity_vals = (img.get_data() > thresh).sum(axis=0).sum(axis=0).sum(axis=0)
                elif 'neg' in s:
                    sparsity_vals = (img.get_data() < -thresh).sum(axis=0).sum(axis=0).sum(axis=0)
                elif 'abs' in s:
                    sparsity_vals = (abs(img.get_data()) > thresh).sum(axis=0).sum(axis=0).sum(axis=0)   
                sparsity[k][s].append(sparsity_vals)
          
         
        # 3) get hpi values for wb components
            if k == "wb":
                hemi_vectors = [masker.transform(img) for masker in hemi_maskers]
                hemi_imgs = [masker.inverse_transform(vec) for masker, vec in     # transform back so that values for each 
                            zip(hemi_maskers, hemi_vectors)]                      # component can be calculated
                         
                pos_vals = [(hemi_img.get_data() > 0.005).sum(axis=0).sum(axis=0).sum(axis=0)        # pos_vals[0] = # voxels in R
                           for hemi_img in hemi_imgs]                                                # pos_vals[1] = # voxels in L
                   
                neg_vals = [(hemi_img.get_data() < -0.005).sum(axis=0).sum(axis=0).sum(axis=0) 
                           for hemi_img in hemi_imgs]
                
                for sign, val in zip(['pos','neg'],[pos_vals, neg_vals]):
                    with np.errstate(divide="ignore", invalid="ignore"):         
                        hpi = (val[0].astype(float) - val[1])/(val[0] + val[1])     # pos/neg HPI vals for each component
                        hpi_mean = np.mean(hpi[np.isfinite(hpi)]) 
                        hpi_sd = np.std(hpi[np.isfinite(hpi)]) 
                        hpi[~np.isfinite(hpi)] = 0                                              # set non finite values to 0
    
                    y_vals[sign]['vals'].append(hpi)
                    y_vals[sign]['mean'].append(hpi_mean)
                    y_vals[sign]['sd'].append(hpi_sd)
        

    # Now plot:
    # 2)
    fh, axes= plt.subplots(1, 3, sharex=True, sharey=True, figsize=(18,6))
    sparsity_styles = {'pos_005':['b','lightblue'],'neg_005':['r','lightpink'], 'abs_005':['g','lightgreen']}
    for ax, k in zip(axes, images_key):
        for s in sparsity_levels:
            mean = np.asarray([np.mean(arr) for arr in sparsity[k][s]])
            sd = np.asarray([np.std(arr) for arr in sparsity[k][s]])
            ax.fill_between(components, mean+sd, mean-sd, linewidth=0, facecolor = sparsity_styles[s][1], alpha=0.5)
            ax.plot(components, mean, color=sparsity_styles[s][0], label=s)
        # Overlay individual points for absolute threshold
        ax.scatter(np.hstack(x), np.hstack(sparsity[k]['abs_005']), c=sparsity_styles['abs_005'][0])
        ax.set_title("Sparsity of the %s components"%(k))
        ax.set_xlim(xmin = components[0]-1, xmax = components[-1]+1)
        ax.set_xticks(components)
    plt.legend()
    fh.text(0.5, 0.04, "# of components", ha="center")
    fh.text(0.04, 0.5, "# of voxels above the threshold", va='center', rotation='vertical')
    
    out_path = op.join(out_dir, 'sparsity.png')
    save_and_close(out_path, fh=fh)
    
    # 3)
    fh, axes = plt.subplots(1, 2, sharex=True, sharey =True, figsize=(12,6))
    fh.suptitle("Hemispheric Participation Index for each component", fontsize=16)
    hpi_styles = {'pos':['b','lightblue', 'above 0.005'], 'neg':['r','lightpink','below -0.005']}
    for ax, sign in zip(axes, ['pos','neg']):
        mean, sd = np.asarray(y_vals[sign]['mean']), np.asarray(y_vals[sign]['sd'])
        ax.fill_between(components, mean+sd, mean-sd, linewidth=0, facecolor = hpi_styles[sign][1], alpha=0.5)
        size = sparsity['wb']['%s_005'%(sign)]
        ax.scatter(np.hstack(x), np.hstack(y_vals[sign]['vals']), label=sign, c=hpi_styles[sign][0], s=np.hstack(size)/20)
        ax.plot(components, mean, c=hpi_styles[sign][0])
        ax.set_title("%s"%(sign))
        ax.set_xlim((0, components[-1]+5))
        ax.set_ylim((-1, 1))
        ax.set_xticks(components)
        ax.set_ylabel("HPI((R-L)/(R+L) for # of voxels %s"%(hpi_styles[sign][2]))
    
    fh.text(0.5, 0.04, "# of components", ha="center")
    
    out_path = op.join(out_dir, 'wb_HPI.png')
    save_and_close(out_path, fh=fh)

if __name__ == '__main__':
    import warnings
    from argparse import ArgumentParser

    # Look for image computation errors
    warnings.simplefilter('ignore', DeprecationWarning)
    warnings.simplefilter('error', RuntimeWarning)  # Detect bad NV images

    # Arg parsing
    hemi_choices = ['R', 'L', 'wb']
    parser = ArgumentParser(description="Really?")
    #parser.add_argument('key1', nargs='?', default='R', choices=hemi_choices)
    #parser.add_argument('key2', nargs='?', default='L', choices=hemi_choices)
    parser.add_argument('--noSimScore', action='store_true', default=False)
    parser.add_argument('--force', action='store_true', default=False)
    parser.add_argument('--offline', action='store_true', default=False)
    parser.add_argument('--components', nargs='?',
                        default="5,10,15,20,25,30,35,40,45,50")
    parser.add_argument('--dataset', nargs='?', default='neurovault',
                        choices=['neurovault', 'abide', 'nyu'])
    parser.add_argument('--seed', nargs='?', type=int, default=42,
                        dest='random_state')
    parser.add_argument('--scoring', nargs='?', default='l1norm',
                        choices=['l1norm', 'l2norm', 'correlation'])
    args = vars(parser.parse_args())

    # Alias args
    query_server = not args.pop('offline')
    #keys = args.pop('key1'), args.pop('key2')
    components = [int(c) for c in args.pop('components').split(',')]

    # Run loops
    do_ic_loop(query_server=query_server,
               components=components, **args)

    plt.show()

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
import scipy.stats as ss

from main import main
from nibabel_ext import NiftiImageWithTerms
from nilearn_ext.masking import HemisphereMasker
from nilearn_ext.plotting import save_and_close
from sklearn.externals.joblib import Memory


def do_ic_loop(components, scoring, dataset, noSimScore=False,
               memory = Memory(cachedir='nilearn_cache'), **kwargs):
    score_mats = []
    wb_images = []
    out_dir = op.join('ica_imgs', dataset)
    for c in components:
        if noSimScore: # only calculate sparsity and HPI using ICA images (i.e. do not run main.py)
            print("Simply loading wb component image for n_component = %s"% c)
            nii_dir = op.join('ica_nii', dataset, str(c))
            img_path = op.join(nii_dir, 'wb_ica_components.nii.gz')
            img = NiftiImageWithTerms.from_filename(img_path)
            wb_images.append(img)
        else:
            print("Running analysis with %d components." % c)
            img_d, score_mats_d, sign_mat_d = main(n_components=c, dataset=dataset,
                                        scoring=scoring, **kwargs)
            score_mats.append(score_mats_d)
            wb_images.append(img_d['wb'])

    # Now we have all the scores; now get values of interest.
    # 1) mean dissimilarity scores for the best match pairs
    sim_scores = None if noSimScore else {key:[] for key in score_mats[0]}
    
    # 2) overall sparsity of the wb components
    sparsity_levels = {'pos_01':0.01,'pos_005':0.005,'neg_01':-0.01,'neg_005':-0.005}
    sparsity_styles = {'pos_01':('b','-'),'pos_005':('b','--'),'neg_01':('r','-'),'neg_005':('r','--')}
    wb_sparsity = {s:([],[]) for s in sparsity_levels}  # store mean and SEM
    
    # 3) hemispheric participation index (HPI): (R-L)/(R+L) for # of voxels>0.005
    # calculated from wb components  
    hemi_maskers = [HemisphereMasker(hemisphere=hemi, memory=memory).fit() for hemi in ['R','L']]
    x = [[c]*c for c in components]
    y1, y2, size1, size2 = [], [], [], []
    
    if not noSimScore:
        # Plot 1)
        for sm_d in score_mats:
        
            for key in sim_scores:
                sm = sm_d[key]
                sm[sm == 0] = np.inf  # is this necessary??
                sim_scores[key].append(sm.min(axis=1).mean())
        
        fh = plt.figure(figsize=(10, 10))
        fh.gca().plot(components, np.asarray(sim_scores.values()).T)
        fh.gca().legend(sim_scores.keys())
        plt.title("Average dissimilarity scores for the best-match pairs")
        plt.xlabel("# of components"), plt.ylabel("mean %s scores" % scoring)

        out_path = op.join(out_dir, '%s_simscores.png' % scoring)
        save_and_close(out_path, fh=fh)
        
    for img in wb_images:    
        # 2) get mean sparsity per component
        for key in wb_sparsity:
            # sparsity_values is a list containing # of voxels above the given sparsity level for each component
            if 'pos' in key:
                sparsity_values = (img.get_data() > sparsity_levels[key]).sum(axis=0).sum(axis=0).sum(axis=0)
            elif 'neg' in key:
                sparsity_values = (img.get_data() < sparsity_levels[key]).sum(axis=0).sum(axis=0).sum(axis=0)
            mean_sparsity = sparsity_values.mean()
            sem = ss.sem(sparsity_values)
            wb_sparsity[key][0].append(mean_sparsity)
            wb_sparsity[key][1].append(sem)
         
        # 3) get hpi values
        hemi_vectors = [masker.transform(img) for masker in hemi_maskers]
        hemi_imgs = [masker.inverse_transform(vec) for masker, vec in     # transform back so that values for each 
                    zip(hemi_maskers, hemi_vectors)]                      # component can be calculated
                         
        pos_vals = [(hemi_img.get_data() > 0.005).sum(axis=0).sum(axis=0).sum(axis=0)        # pos_vals[0] = # voxels in R
                   for hemi_img in hemi_imgs]                                                # pos_vals[1] = # voxels in L
                   
        neg_vals = [(hemi_img.get_data() < -0.005).sum(axis=0).sum(axis=0).sum(axis=0) 
                   for hemi_img in hemi_imgs]
        
        with np.errstate(divide="ignore", invalid="ignore"):         
            (pos_hpi, neg_hpi) = [(vals[0].astype(float) - vals[1])/(vals[0] + vals[1])     # pos/neg HPI vals for each component
                                for vals in [pos_vals, neg_vals]]  
        pos_hpi[~np.isfinite(pos_hpi)] = 0                                              # set non finite values to 0
        neg_hpi[~np.isfinite(neg_hpi)] = 0 
        y1.append(pos_hpi)
        y2.append(neg_hpi)
        
        (pos_size, neg_size) = [vals[0]+vals[1] for vals in [pos_vals, neg_vals]]    
        size1.append(pos_size)
        size2.append(neg_size)
    
    # Now plot:
    # 2)
    fh = plt.figure(figsize=(10, 10))
    for key in wb_sparsity:
        mean, sem = wb_sparsity[key][0], wb_sparsity[key][1]
        color, linestyle = sparsity_styles[key][0], sparsity_styles[key][1]
        plt.errorbar(components, mean, sem, color=color, linestyle=linestyle, label=key)
    plt.legend()
    plt.xlim(xmax = components[-1]+1)
    plt.xticks(components)
    plt.xlabel("# of components"), plt.ylabel("mean # of voxels above the threshold")
    plt.title("Average sparsity of the wb components")
    
    out_path = op.join(out_dir, 'wb_sparsity.png')
    save_and_close(out_path, fh=fh)
    
    # 3)
    fh = plt.figure(figsize=(12,10))  
    plt.scatter(np.hstack(x)+1, np.hstack(y1), label="positive", c="b", s=np.hstack(size1)/10)
    plt.scatter(np.hstack(x)-1, np.hstack(y2), label="negative", c="r", s=np.hstack(size2)/10)
    
    plt.legend(bbox_to_anchor=(1.1,1))
    plt.xlim(xmax = components[-1]+5), plt.ylim((-1, 1))
    plt.xticks(components)
    plt.xlabel("# of components"), plt.ylabel("HPI((R-L)/(R+L) for # of voxels above 0.005")
    plt.title("Hemispheric Participation Index for each component")
    
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

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.utils import resample

def _argparse():
    import argparse
    parser = argparse.ArgumentParser(description='Choose csv and define lists to analyze')
    parser.add_argument('--csvpath', type=str, default='./docs/source_analysis_cleansed.csv', help='CSVpath to analyze')
    parser.add_argument('--savedir', type=str, default='./outputs/metrics', help='Save dir')
    parser.add_argument('--target', nargs='*', required=True, help='Target columns. ex) --target AS MR')
    parser.add_argument('--pred', type=str, default='pred_label_', help='Preffix for predicted target')
    parser.add_argument('--true', type=str, default='label_', help='Preffix for label target')
    parser.add_argument('--color', type=str, default='crimson', choices=['crimson', 'darkblue', 'darkgreen', 'goldenrod'], help='Color for ROC')
    parser.add_argument('--boot', type=int, default=1000, help='Bootstrap number')
    parser.add_argument('--ci_cutoff', type=str, default='95', choices=['95', '99'], help='Bootstrap number')
    args = parser.parse_args()
    return args

def _roc_base():
    '''Make an ROC base.'''
    fig, ax = plt.subplots(figsize=(5,5), dpi=300)
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
    plt.xticks(np.arange(0, 1.05, step=0.1))
    plt.yticks(np.arange(0, 1.05, step=0.1))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.plot([0, 1], [0, 1], linestyle="--", lw=0.25, color="gray", alpha=0.8)    
    return fig, ax 

def roc_auc(s_y, s_x, color):
    '''Make a plain ROC.
    Returned value is a figure of the ROC.'''
    fpr, tpr, thresholds = metrics.roc_curve(s_y, s_x)
    fig, ax = _roc_base()
    plt.plot(fpr, tpr, marker=None, color=color, lw=0.75)
    return fig

def roc_auc_with_CI(s_y, s_x, color, boot, ci_cutoff):
    '''Make an ROC with CI using the bootstrapping.
    Returned values are auc and a figure of the roc with 95%CI.'''

    fpr, tpr, thresholds = metrics.roc_curve(s_y, s_x)
    auc_type_divide = metrics.auc(fpr, tpr)
    fig, ax = _roc_base()
    plt.plot(fpr, tpr, marker=None, color=color, lw=0.75)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    i = boot
    for i in range(i):
        resampled_y, resampled_x = resample(s_y, s_x)
        resampled_fpr, resampled_tpr, resampled_thresholds = metrics.roc_curve(resampled_y, resampled_x)
        resampled_roc_auc = metrics.auc(resampled_fpr, resampled_tpr)
        interp_tpr = np.interp(mean_fpr, resampled_fpr, resampled_tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(resampled_roc_auc)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0

    std_tpr = np.std(tprs, axis=0)
    if args.ci_cutoff == '95':
        tprs_upper = np.minimum(mean_tpr + 1.96 * std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - 1.96 * std_tpr, 0)
        auc_conf_interval = np.percentile(aucs,[2.5,97.5])
    elif args.ci_cutoff == '99':
        tprs_upper = np.minimum(mean_tpr + 2.575 * std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - 2.575 * std_tpr, 0)  
        auc_conf_interval = np.percentile(aucs,[0.5,99.5])      
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color=color, lw=0.25, alpha=0.1, label=False)

    auc_with_ci = f'{auc_type_divide:.2f}({auc_conf_interval[0]:.2f}-{auc_conf_interval[1]:.2f})'
    s_auc = pd.Series([auc_with_ci])
    return fig, s_auc

def main(args):
    df = pd.read_csv(args.csvpath)
    s_auc = pd.Series(dtype='object', name='auc')
    for institution in df['Institution'].unique():
        df_institution = df.query('Institution == @institution')
        d_target = {args.true + i: args.pred + i for i in args.target}
        for split in df_institution['split'].unique():
            df_split = df_institution.query('split == @split')
            for k,v in d_target.items():
                s_y = df_split[k]
                s_x = df_split[v]
                roc_plain = roc_auc(s_y, s_x, args.color)
                roc_with_ci, s_tmp = roc_auc_with_CI(s_y, s_x, args.color, args.boot, args.ci_cutoff)
                roc_savedir = os.path.join(args.savedir, 'imgs', institution, split)
                os.makedirs(roc_savedir, exist_ok=True)
                roc_plain.savefig(roc_savedir + '/roc_plain_' + k + '.png', transparent=True)
                roc_with_ci.savefig(roc_savedir + '/roc_with_ci_' + k + '.png', transparent=True)
                s_tmp.index = ['roc_' + institution + '_' + split + '_' + k]
                s_auc = pd.concat([s_auc, s_tmp])      
    s_auc.to_csv(args.savedir + '/roc.csv')  

if __name__ == '__main__':
    args = _argparse()
    main(args)

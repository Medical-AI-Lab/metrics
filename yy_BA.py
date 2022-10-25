import math
import os
import numpy as np
import pandas as pd
import scipy
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn import metrics

def _argparse():
    import argparse
    parser = argparse.ArgumentParser(description='Choose csv and define lists to analyze')
    parser.add_argument('--csvpath', type=str, default='./docs/source_analysis_cleansed.csv', help='CSVpath to analyze')
    parser.add_argument('--savedir', type=str, default='./outputs/metrics', help='Save dir')
    parser.add_argument('--target', nargs='*', required=True, help='Target columns. ex) --target FEV1 FVC')
    parser.add_argument('--pred', type=str, default='pred_label_', help='Preffix for predicted target')
    parser.add_argument('--true', type=str, default='label_', help='Preffix for label target')
    parser.add_argument('--color', type=str, default='crimson', choices=['crimson', 'darkblue', 'darkgreen', 'goldenrod'], help='Color for figures')
    parser.add_argument('--markersize', type=int, default=1, help='Dots size')
    parser.add_argument('--diffmax', default=False, type=str, help='Set maximam value as round int')
    parser.add_argument('--ci', type=int, default=None, help='Make CI or not')
    parser.add_argument('--boot', type=int, default=1000, help='Dots size')
    parser.add_argument('--randomline', default=False, help='CI')
    parser.add_argument('--min_0', default=False, help='Minimum value in the figure.')

    args = parser.parse_args()
    return args

def yyplot(s_real, s_fake, color, markersize, ci, boot, randomline, min_0):
    fig, ax = plt.subplots(figsize=(9, 9), dpi=300)
    max_ax = np.max([s_real.max(), s_fake.max()])
    if min_0:
        plt.xlim(0, max_ax)
        plt.ylim(0, max_ax)
    if randomline:
        ax.plot([0, max_ax], [0, max_ax], linestyle="--", lw=1, color="gray", alpha=0.8)
    sns.regplot(x=s_real, y=s_fake, ci=ci, n_boot=boot, truncate=False, scatter_kws={"marker" :".", "color": color, "s":markersize}, line_kws={"color": color,"linewidth": 1.5})  
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    return fig 

def fetchmaxdiff(df, k, v):
    maxvalue = 0
    s_diff = abs(df[k]-df[v])
    max_diff = s_diff.max()
    if max_diff >= maxvalue:
        maxvalue = max_diff
    return maxvalue

def BAplot(s_label, s_pred, color, markersize, maxvalue):
    fig, ax = plt.subplots(1, figsize=(9, 9), dpi=300)
    sm.graphics.mean_diff_plot(s_label, s_pred, ax=ax, scatter_kwds={"marker" :".", "color": color, "s":markersize})
    plt.ylim(-maxvalue, maxvalue)
    plt.close()
    return fig

def main(args):
    df = pd.read_csv(args.csvpath)
    d_target = {args.true + i: args.pred + i for i in args.target}
    for k,v in d_target.items():
        df_target = df[[k, v]]
        maxvalue = fetchmaxdiff(df_target, k, v) #Fetch maxvalue to make consistent BA figures.
        for institution in df['Institution'].unique():
            df_institution = df.query('Institution == @institution')
            for split in df_institution['split'].unique():
                df_split = df_institution.query('split == @split')
                s_y = df_split[k]
                s_x = df_split[v]
                yy_img = yyplot(s_y, s_x, args.color, args.markersize, args.ci, args.boot, args.randomline, args.min_0)
                BA_img = BAplot(s_y, s_x, args.color, args.markersize, maxvalue)
                img_savedir = os.path.join(args.savedir, 'imgs', institution, split)
                os.makedirs(img_savedir, exist_ok=True)
                yy_img.savefig(img_savedir + '/yyplot_' + k + '.png', transparent=True)
                BA_img.savefig(img_savedir + '/baplot_' + k + '.png', transparent=True)

if __name__ == '__main__':
    args = _argparse()
    main(args)

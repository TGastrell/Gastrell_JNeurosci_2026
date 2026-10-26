#%%
'''Import Libraries, load data into one big dataframe, drop bad_trials, round off kappa values and compute errors
'''
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from analysis.common import utils
from analysis.common.util_dataloader import load_beh_data
from analysis.common.paths import experiment_rawdata_root, figure_output_path
from analysis.behaviour.plotting_helpers import get_cmean_summary, get_mean_summary

TICK_LABEL_SIZE = 18
AX_LABEL_SIZE = 20
PANEL_LABEL_SIZE = 20

def plot_effects(fit, ax, ylabels = [r'$\kappa_s$', r'$\kappa_p$', r'$\kappa_s$ x $\kappa_p$']):
    ax_ = ax.inset_axes([0.75, 0.8, 0.15, 0.15])  # [left, bottom, width, height]
    ax_.set_ylabel(' ')
    ax_.set_xlabel('t-value', fontsize=AX_LABEL_SIZE-2)
    sns.barplot(x = fit.tvalues.values[1:4], 
                y = fit.tvalues.index[1:4], 
                color='lightgrey', 
                ax=ax_)
    
    ax_.set_yticklabels(ylabels, fontsize=TICK_LABEL_SIZE-2)
    #ax_.set_xticklabels(ax_.get_xticks(), fontsize=TICK_LABEL_SIZE-4)
    ax_.tick_params(axis='both', which='major', labelsize=TICK_LABEL_SIZE-4)

    sns.despine(trim=True, ax=ax_)
    ax_.axvline(x=0, color='grey', linestyle='dashed')

    # plot significant asterisks
    pos = max(fit.tvalues.values[1:4]) + 0.2
    for i, pval in enumerate(fit.pvalues.values[1:4]):
        # center the asterisks on the bars
        if pval < 0.05:
            ax_.text(pos, i, '*', fontsize=14, va='center')
        elif pval < 0.01:
            ax_.text(pos, i, '**', fontsize=14, va='center')


def make_bhv_quad_figs(df):
    '''
    Generate behavioural figures for the manuscript.
    Args:
        df (pd.DataFrame): The DataFrame containing the behavioural data.
        save_figs (bool): Whether to save the generated figures. Default is False.
    Returns:
        None
    '''
    colors = sns.color_palette("Blues", n_colors=3)

    bias_ylim = (-0.3, 0.5)
    prec_ylim = (0.6,1.2)
    hist_ylim = (0, 100)
    fig= plt.figure(figsize=(17,17))

    #=================================Histograms======================================
    n_bins = 30
    ax = fig.add_subplot(2,2,1,projection='3d' )

    for i, k_stim in enumerate(sorted(df.stimKappa.unique())):
            df_subset = df.query('stimKappa == @k_stim')
            hist, bin_edges = np.histogram(df_subset.stim_response_error, bins=n_bins, range=(-np.pi, np.pi))
            xs = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            ys = hist
            
            # offset each histogram in z direction
            zs = i * 50

            ax.bar(xs, ys, zs=zs, zdir='y', alpha=1, width=(bin_edges[1]-bin_edges[0]), color=colors[i], label=rf'$\kappa_s$={k_stim}', ec = 'black')


    ax.grid(False)
    ax.set_ylim(hist_ylim)
    ax.set(zlim=( 0, 8000),
        zticklabels=[],
        zticks=[],
        yticks=[])
    ax.set_yticklabels('')
    ax.set_xticks([-np.pi, 0, np.pi])
    ax.set_xticklabels([r'-$\pi$', '0', r'$\pi$'], fontsize=TICK_LABEL_SIZE)
    #ax.set_xlabel('Error Magnitudes',fontsize=AX_LABEL_SIZE)
    ax.set_title('Error Magnitudes',fontsize=PANEL_LABEL_SIZE, fontweight='bold')
    #ax.text(-3.2, 12000, 'A)', fontsize=16, fontweight='bold', va='top')
    ax.legend(title='Stimulus Precision', bbox_to_anchor=(0.3, 0.85), fontsize=14, title_fontsize=14)
    #==================================Accuracy/Precision====================================
    summary_cmean = get_mean_summary(df, ['subjectNumber', 'priorKappa', 'stimKappa', 'accuracy'])
    summary_cmean['cmean_norm'] = np.clip(summary_cmean['cmean_norm'], 0, 1)  # Clip values to be between 0 and 1 since there seems to be some floating point error causing values slightly above 1 or below 0. This ensures the boxplot remains within the expected range.
    ax = fig.add_subplot(2,2,2)
    sns.boxplot(data = summary_cmean,
                    x='priorKappa',
                    y='cmean_norm',
                    hue = 'stimKappa',
                    palette=colors,
                    width=0.25,
                    #dodge=True,
                    legend=False,
                    ax=ax)

    ax.set_ylabel(" 1 - Absolute Error Magnitude", fontsize=AX_LABEL_SIZE)
    ax.set_xlabel(r"Prior Precision $\kappa_p$", fontsize=AX_LABEL_SIZE)
    ax.set_title('Response Accuracy',fontsize=PANEL_LABEL_SIZE, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_SIZE)
    ax.set_ylim(prec_ylim)

    # inset axes for regression weights

    model = smf.mixedlm('accuracy ~ stimKappa * priorKappa', 
                        data = df, 
                        groups = df.subjectNumber)
    plot_effects(model.fit(), ax)
    #================================Bias=======================================
    summary_cmean = get_cmean_summary(df, ['subjectNumber', 'priorKappa', 'stimKappa', 'signed_error'])

    ax = fig.add_subplot(2,2,3)
    ax.axhline(y = 0, 
        c='red', 
        linestyle='dashed', 
        alpha = 0.6)
    sns.boxplot(data = summary_cmean, 
                    x='priorKappa',
                    y='cmean_norm',
                    hue = 'stimKappa',
                    palette=colors,
                    width=0.25,
                    #dodge=True,
                    ax=ax,
                    legend=False)
    
    
    ax.set_ylabel("Signed Error Magnitude", fontsize=AX_LABEL_SIZE)
    ax.set_xlabel(r"Prior Precision $\kappa_p$", fontsize=AX_LABEL_SIZE)
    ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_SIZE)
    ax.set_ylim(bias_ylim)

    ax.set_title('Response Bias',fontsize=PANEL_LABEL_SIZE, fontweight='bold')
    # inset axes for regression weights

    model = smf.mixedlm('signed_error ~ stimKappa * priorKappa', 
                        data = df, 
                        groups = df.subjectNumber)
    plot_effects(model.fit(), ax)

    #===============================Bias over Time========================================
    summary_cmean = get_cmean_summary(df, ['subjectNumber', 't_across','stimKappa', 'signed_error'])
    summary_cmean['t_across'] = summary_cmean['t_across'].cat.remove_unused_categories()
    colors = sns.color_palette("Blues", n_colors=3)
    ax = fig.add_subplot(2,2,4)
    ax.axhline(y = 0, 
        c='red', 
        linestyle='dashed', 
        alpha = 0.6)
    sns.boxplot(data = summary_cmean,
                    x='t_across',
                    y='cmean_norm',
                    hue = 'stimKappa',
                    palette=colors,
                    width=0.25,
                    #dodge=True,
                    ax=ax,
                    legend=False)
    ax.set_title('Response Bias Over Time',fontsize=PANEL_LABEL_SIZE, fontweight='bold')
    ax.set_ylim(bias_ylim)
    ax.set_ylabel("Signed Error Magnitude", fontsize=AX_LABEL_SIZE)
    ax.set_xlabel("Time (Blocks)", fontsize=AX_LABEL_SIZE)
    ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_SIZE)
    #ax.legend(title='Time', bbox_to_anchor=(1.02, 0.2))
    # inset axes for regression weights
    for_model = df.copy()
    for_model['t_across'] = for_model['t_across'].cat.codes  # Convert to numeric codes
    model = smf.mixedlm('signed_error ~ stimKappa * t_across', 
                        data = for_model, 
                        groups = for_model.subjectNumber)
    plot_effects(model.fit(), 
                ax,
                [r'$\kappa_s$', 
                'Time', 
                r'$\kappa_s$ x Time']
                )

    sns.despine(trim=True)
    fig.text(0.1, 0.88, 'A', fontsize=PANEL_LABEL_SIZE, fontweight='bold', va='top', ha='left')
    fig.text(0.52, 0.88, 'B', fontsize=PANEL_LABEL_SIZE, fontweight='bold', va='top', ha='left')
    fig.text(0.1, 0.50, 'C', fontsize=PANEL_LABEL_SIZE, fontweight='bold', va='top', ha='left')
    fig.text(0.52, 0.50, 'D', fontsize=PANEL_LABEL_SIZE, fontweight='bold', va='top', ha='left')
    fig.tight_layout(h_pad=5, w_pad=5) 



def main():
    for experiment in (1, 2):
        rawdata = experiment_rawdata_root(experiment)
        df, _ = load_beh_data(rawdata, bads=True)

        df['stim_response_error'] = utils.circular_difference(
            df.responseAngle, df.stimDir
        )
        df['prior_stim_distance'] = utils.circular_difference(
            df.priorDir, df.stimDir
        )
        df['signed_error'] = utils.get_signed_error(
            df.stim_response_error, df.prior_stim_distance
        )
        df['t_across'] = pd.qcut(
            df['blockNr'], 3, labels=['Early', 'Middle', 'Late']
        )
        df['accuracy'] = (
            1 - np.abs(df['stim_response_error']) / np.pi
        )  # accuracy is defined as 1 - normalized absolute error

        make_bhv_quad_figs(df)
        plt.savefig(
            figure_output_path(f'E{experiment}_bhv_results.svg'),
            dpi=300,
            bbox_inches='tight',
        )


if __name__ == '__main__':
    main()

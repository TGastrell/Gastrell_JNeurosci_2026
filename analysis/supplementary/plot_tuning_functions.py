#%%
import mne
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from analysis.supplementary import IEM
import seaborn as sns
from analysis.common.paths import (
    WITHIN_PRIOR_FE_DIRECTORY,
    analysis_cache_dir,
    derivatives_root,
    figure_output_path,
    within_prior_cache_dir,
)
from analysis.common.util_dataloader import (
    crop_full_window_metrics,
    load_decoding_data,
)

TICK_LABEL_SIZE = 18
AX_LABEL_SIZE = 20
PANEL_LABEL_SIZE = 20

def subject_level_tuning_summary(_df, dv, split_by = 'Coherence'):

    summary = (_df
                .groupby(['subjectNumber', 'channel', split_by])[dv]
                .mean()
                .reset_index()
                )

    summary = summary.merge(
            _df.groupby(['subjectNumber', 'channel'])[dv]
            .mean()
            .reset_index()
            .rename(columns= {dv:'sub_avg_evk' })
    )

    summary= summary.merge(
        summary.groupby(['channel'])[['sub_avg_evk']]
        .mean()
        .reset_index()
        .rename(columns={'sub_avg_evk':'grand_mean'})
    )

    summary['evk_norm'] = summary[dv] - summary['sub_avg_evk'] + summary['grand_mean']
    return summary

def subject_level_acc_summary(_df, dv, split_by = 'Coherence'):
    summary = (_df
                .groupby(['subjectNumber', split_by])[dv]
                .mean()
                .reset_index()
                )

    summary = summary.merge(
            _df.groupby(['subjectNumber',])[dv]
            .mean()
            .reset_index()
            .rename(columns= {dv:'mean' })
    )

    grand_mean = summary['mean'].mean()

    summary['mean_norm'] = summary[dv] - summary['mean'] + grand_mean
    return summary

def plot_aggregate_acc(ax, df, split_by, palette=None):
    if palette is None:
        palette = sns.color_palette("Blues", n_colors=3)

    ax_ = ax.inset_axes([0.75, 0.6, 0.25, 0.25])  # [left, bottom, width, height]

    sns.barplot(df,
                x = split_by, 
                y = 'mean_norm', 
                palette=palette,
                errorbar='se', 
                ax=ax_)
    ax_.set_yticks([0, 0.1, 0.2])
    ax_.set_yticklabels(['0', '0.1', '0.2'], fontsize=TICK_LABEL_SIZE-2)
    ax_.set_ylabel('', fontsize=AX_LABEL_SIZE)
    ax_.set_xlabel('', fontsize=AX_LABEL_SIZE)
    ax_.set_xticklabels('')
    ax_.set_ylim(0,0.2)
    #ax.tick_params(axis='both', which='major', labelsize=14)
    sns.despine(trim=True, ax=ax_)


def get_tuning_funcs(epochs, time_origin):
    stims = epochs.metadata['stimDir'].values  # shape (n_trials,)
    forward_model = IEM.get_forward_model(stims)
    aligned = IEM.align_C_test(epochs.get_data(), stims, forward_model).astype('float32')

    epochs._data = aligned
    df = pd.merge(
        epochs.to_data_frame(long_format=True, time_format=None),
        epochs.metadata,
        left_on='epoch',
        right_index=True,
    )
    channel_angles = dict(zip(epochs.ch_names, np.linspace(-np.pi, np.pi, 16, endpoint=False).round(2)))
    df['channel'] = df['channel'].map(channel_angles)
    df['channel'] = df['channel'].astype('float')
    df['time'] = df['time'] - time_origin
    df = df.query('time > -0.25 & time < 2')
    return (df[['subjectNumber','time', 'channel','stimKappa', 'priorKappa', 'value']]
            .groupby(['subjectNumber','time', 'channel','stimKappa', 'priorKappa'])
            .mean()
            .reset_index())
#%%
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate the main-model (S1) or within-prior (S4) tuning figure.'
    )
    parser.add_argument(
        '--analysis',
        choices=('main', 'within-prior'),
        default='within-prior',
        help='decoder mode to use (default: within-prior)',
    )
    args = parser.parse_args()

    derivatives = derivatives_root()
    within_prior = args.analysis == 'within-prior'

    if within_prior:
        tf_fname = 'allsub_IEM_train_within_resamp_binned_channel_activities_long_t--0.25-2_bl--0.35--0.25-0.2-40bpf-clean_cropped_from_full_window.parquet'
        file_label='t--0.95-2_bl--0.35--0.25-0.2-40bpf-clean'
        metric_time_origin = 0.25
        metric_cache_label = 't--0.25-2_bl--0.35--0.25-0.2-40bpf-clean_cropped_from_full_window'
        metric_pth = within_prior_cache_dir() / f'trial_avg_decoding_metrics_resamp_binned_long_{metric_cache_label}.parquet'
        fe_subdirectory = WITHIN_PRIOR_FE_DIRECTORY
    else:
        tf_fname = 'allsub_IEM_channel_activities_long_t--0.25-2_bl--0.35--0.25-0.2-40bpf-clean.parquet'
        file_label='t--0.25-2_bl--0.35--0.25-0.2-40bpf-clean'
        time_origin = 0.25
        metric_time_origin = 0.25
        metric_pth = analysis_cache_dir() / f'trial_avg_decoding_metrics_long_{file_label}.parquet'
        fe_subdirectory = 'train_sk2_unif'

    if metric_pth.exists():
        metrics = pd.read_parquet(metric_pth)
    else:
        metrics = load_decoding_data(
            derivatives,
            file_label,
            ['subjectNumber', 'time', 'stimKappa', 'priorKappa', 'accuracy'],
            subdirectory=fe_subdirectory,
        )
        if within_prior:
            metrics = crop_full_window_metrics(metrics)
        metrics = (
            metrics.groupby(
                ['subjectNumber', 'time', 'stimKappa', 'priorKappa']
            )['accuracy']
            .mean()
            .reset_index()
        )

    metrics['time_secs'] = metrics['time'] / 256 - metric_time_origin
    metrics.to_parquet(metric_pth, index=False)
    metrics['time'] = metrics['time_secs']

    tf_path = (within_prior_cache_dir() if within_prior else analysis_cache_dir()) / tf_fname
    if tf_path.exists():
        df = pd.read_parquet(tf_path)
    else:
        all_pts = []
        sorted_pts = sorted(derivatives.iterdir(), key=lambda x: x.name.lower())
        for sub_dir in sorted_pts:
            if not (sub_dir.stem.startswith('sub') and sub_dir.is_dir()):
                continue    

            sub = sub_dir.stem

            FE_dir = sub_dir / 'eeg' / 'FE' / fe_subdirectory

            # load predicted channel activations
            epochs_file = FE_dir / f'{sub}_task-TG-MotEst-2.0_{file_label}_FE-pred-epochs.fif.gz'

            if not epochs_file.exists():
                print(f'File not found: {epochs_file}')
                continue
            
            epochs = mne.read_epochs(epochs_file, preload=True)
            n_trials, n_chans, n_time = epochs.get_data().shape

            if within_prior:
                epochs.crop(tmin=-0.25, tmax=2)
                activity_time_origin = 0.25
            else:
                activity_time_origin = time_origin
            all_pts.append(get_tuning_funcs(epochs, activity_time_origin))
            del epochs  # free memory

        df = pd.concat(all_pts, ignore_index=True)
        df.to_parquet(tf_path)


    metrics_summary = (metrics[['subjectNumber','time','stimKappa', 'priorKappa', 'accuracy']]
            .groupby(['subjectNumber','time', 'stimKappa', 'priorKappa'])
            .mean()
            .reset_index())
    

    stim_colors = sns.color_palette("Blues", n_colors=3)
    prior_colors = sns.color_palette("Oranges", n_colors=3)

    fig, axes = plt.subplots(2, 2, figsize=(15,9))
    stim_summary = subject_level_tuning_summary(df.query('time > 0.25 & time < 0.75'), 
                                        split_by='stimKappa', 
                                        dv='value')
    sns.lineplot(stim_summary,
                x='channel',
                y='evk_norm',
                hue='stimKappa',
                palette=stim_colors,
                ax=axes[0,0],
                legend=False)
    axes[0,0].set_ylim(-0.06, 0.15)
    axes[0,0].set_title('Observation Period (0.25-0.75)', fontsize=PANEL_LABEL_SIZE)
    axes[0,0].set_ylabel('Channel Activity', fontsize=AX_LABEL_SIZE)
    axes[0,0].set_xlabel('')

    stim_metric_summary = subject_level_acc_summary(metrics_summary.query('time > 0.25 & time < 0.75'), 
                                        split_by='stimKappa', 
                                        dv='accuracy')

    plot_aggregate_acc(axes[0,0], stim_metric_summary, split_by='stimKappa')
    ####################################################################################

    stim_summary = subject_level_tuning_summary(df.query('time > 1 & time < 1.5'), 
                                        split_by='stimKappa', 
                                        dv='value')

    sns.lineplot(stim_summary,
                x='channel',
                y='evk_norm',
                hue='stimKappa',
                palette=stim_colors,
                ax=axes[0,1],
                legend=True)
    axes[0,1].set_ylim(-0.18, 0.25)
    axes[0,1].set_title('Response Period (1.0-1.5)', fontsize=PANEL_LABEL_SIZE)
    axes[0,1].set_ylabel('')
    axes[0,1].set_xlabel('')

    axes[0,1].legend(bbox_to_anchor=(1,1.01), title=r'Stimulus Precision ($\kappa_s$)', frameon=False, fontsize=AX_LABEL_SIZE,title_fontsize=PANEL_LABEL_SIZE)

    stim_metric_summary = subject_level_acc_summary(metrics_summary.query('time > 1 & time < 1.5'), 
                                        split_by='stimKappa', 
                                        dv='accuracy')

    plot_aggregate_acc(axes[0,1], stim_metric_summary, split_by='stimKappa')

    ####################################################################################
    prior_summary = subject_level_tuning_summary(df.query('time > 0.25 & time < 0.75'), 
                                        split_by='priorKappa', 
                                        dv='value')
    sns.lineplot(prior_summary,
                x='channel',
                y='evk_norm',
                hue='priorKappa',
                palette=prior_colors,
                ax=axes[1,0],
                legend=False)
    axes[1,0].set_ylim(-0.06, 0.15)
    axes[1,0].set_ylabel('Channel Activity', fontsize=AX_LABEL_SIZE)
    #axes[1,0].set_xlabel('Motion Channel (rad.)', fontsize=AX_LABEL_SIZE)

    #axes[1,0].set_title('Observation Period', fontsize=16)
    prior_metric_summary = subject_level_acc_summary(metrics_summary.query('time > 0.25 & time < 0.75'), 
                                        split_by='priorKappa', 
                                        dv='accuracy')

    plot_aggregate_acc(axes[1,0], prior_metric_summary, split_by='priorKappa', palette=prior_colors)
    #########################################################################################
    prior_summary = subject_level_tuning_summary(df.query('time > 1 & time < 1.5'), 
                                        split_by='priorKappa', 
                                        dv='value')
    sns.lineplot(prior_summary,
                x='channel',
                y='evk_norm',
                hue='priorKappa',
                palette=prior_colors,
                ax=axes[1,1],
                legend=True)
    axes[1,1].set_ylim(-0.18, 0.25)
    #axes[1,1].set_xlabel('Motion Channel (rad.)', fontsize=16)
    axes[1,1].set_ylabel('')

    #axes[1,1].set_title('Response Period',fontsize=16)
    axes[1,1].legend(bbox_to_anchor=(1,1.01),title=r'Prior Precision ($\kappa_p$)', frameon=False, fontsize=AX_LABEL_SIZE,title_fontsize=PANEL_LABEL_SIZE)

    prior_metric_summary = subject_level_acc_summary(metrics_summary.query('time > 1 & time < 1.5'), 
                                        split_by='priorKappa', 
                                        dv='accuracy')

    plot_aggregate_acc(axes[1,1], prior_metric_summary, split_by='priorKappa', palette=prior_colors)


    for ax in axes.flatten():
        ax.set_xlabel('Motion Channel (rad.)', fontsize=AX_LABEL_SIZE)
        ax.tick_params(labelsize=TICK_LABEL_SIZE)

    fig.text(0.01, 1, 'A', fontsize=PANEL_LABEL_SIZE, fontweight='bold', va='top', ha='left')
    fig.text(0.01, 0.5, 'B', fontsize=PANEL_LABEL_SIZE, fontweight='bold', va='top', ha='left')
    sns.despine()

    plt.tight_layout()
    filename = 'figS4.svg' if within_prior else 'figS1.svg'
    plt.savefig(figure_output_path(filename),
                dpi = 300,
                bbox_inches='tight'
                )
# %%

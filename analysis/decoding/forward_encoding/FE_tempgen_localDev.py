#%%
import argparse
import numpy as np
import mne
import logging
from analysis.common import utils
from joblib import Parallel, delayed
from analysis.common.paths import subject_directories

def process_timepoint(ti, tj, train_data, test_data, loocv_means, test_stims, weights, neuron_prefs):
    # Precompute per (ti, tj) - these are scalars or small arrays
    trn_mean_ti_all = loocv_means[:, :, ti]  # Shape: (n_trl, n_elc)
    test_data_tj = test_data[:, :, tj]       # Shape: (n_trl, n_elc)
    test_data_demeaned = test_data_tj - trn_mean_ti_all  # Shape: (n_trl, n_elc)

    weights_ti = weights[:, :, :, ti]  # Shape: (n_trl, n_chan, n_elc)

    # Vectorized prediction: einsum for batched matrix multiplication
    pred = np.einsum('tce,te->tc', weights_ti, test_data_demeaned)  # Shape: (n_trl, n_chan)

    # Vectorized decoding
    exp_prefs = np.exp(neuron_prefs * 1j)  # Shape: (n_chan,) - precompute outside if possible
    theta = np.sum(pred * exp_prefs, axis=1)  # Shape: (n_trl,) - equivalent to np.inner per row
    decoded_angle = np.angle(theta)  # Shape: (n_trl,)
    decoded_error = np.abs(utils.circular_difference(decoded_angle, test_stims))  # Shape: (n_trl,)
    accuracy = np.cos(decoded_error)  # Shape: (n_trl,)

    return np.mean(accuracy)  # Scalar output
    
    # n_trl  = train_data.shape[0]
    # accuracy = []
    # for trial in range(n_trl):
    #     # Use pre-computed mean for this trial at training time ti
    #     trn_mean_ti = loocv_means[trial, :, ti]
        
    #     # Demean the test data using this fold's training mean
    #     test_data_demeaned = test_data[trial, :, tj] - trn_mean_ti
        
    #     # Apply spatial filter
    #     v = weights[trial, :, :, ti]
    #     pred = v @ test_data_demeaned
        
    #     # Decode
    #     theta = np.inner(pred, np.exp(neuron_prefs * 1j))
    #     decoded_angle = np.angle(theta)
    #     decoded_error = np.abs(utils.circular_difference(decoded_angle, test_stims[trial]))
    #     accuracy.append(np.cos(decoded_error))
    # return np.mean(accuracy)
        
        
            
def run_subject(sub_dir):
    tst_pk = [0, 5, 10]
    tst_sk = [0.2, 0.4, 2]

    t_min = -0.95
    t_max = 2
    bl_min = -0.35
    bl_max = -0.25
    bpf_low = 0.2
    bpf_high = 40

    sub = sub_dir.stem

    epoch_dir = sub_dir / 'eeg/'

    epochs_01 = mne.read_epochs(epoch_dir/f'{sub}_ses-MAIN-01_task-TG-MotEst-2.0_t-{t_min}-{t_max}_bl-{bl_min}-{bl_max}-{bpf_low}-{bpf_high}bpf-clean_epo.fif.gz', 
                                preload=True)
    epochs_01.baseline = None
    epochs_02 = mne.read_epochs(epoch_dir/f'{sub}_ses-MAIN-02_task-TG-MotEst-2.0_t-{t_min}-{t_max}_bl-{bl_min}-{bl_max}-{bpf_low}-{bpf_high}bpf-clean_epo.fif.gz', 
                                preload=True)
    epochs_02.baseline = None

    # timepoints for the pre-trained weights! Likely not the same as the full epochs
    t_min = -0.25
    t_max = 2
    epochs = mne.concatenate_epochs([epochs_01, epochs_02]).pick_types(eeg = True).crop(tmin=t_min, tmax=t_max)
    epochs = epochs["abort == 'NO' & responseTime > 0.2"]

    for pk in range(len(tst_pk)):
        for sk in range(len(tst_sk)):

            train_query = 'priorKappa == 0 & stimKappa == 2'
            test_query = f'priorKappa == {tst_pk[pk]} & stimKappa == {tst_sk[sk]}'
            train_epochs_tg = epochs[train_query]

            train_data = epochs[train_query].get_data() # trials x elec x time
            test_data = epochs[test_query].get_data()
            
            n_trl, n_elc, n_time, n_chan = train_data.shape[0], train_data.shape[1], train_data.shape[2], 16

            outputs = np.empty((n_time, n_time))

            test_stims = np.array(epochs[test_query].metadata.reset_index(drop = True)['stimDir'])

            # neuron prefered motion directions (channel centers)
            neuron_prefs = np.linspace(-np.pi, np.pi, n_chan, endpoint=False)

            if (tst_pk[pk] == 0) and (tst_sk[sk] == 2) : # training set case
                weights = np.load(epoch_dir /  # trials x chans x electrocdes x timepoints
                                  'FE' / 
                                  'train_sk2_unif' / 
                                  f'{sub}_task-TG-MotEst-2.0_t-{t_min}-{t_max}_bl-{bl_min}-{bl_max}-{bpf_low}-{bpf_high}bpf-clean_FE-xval-weights.npy')
                accuracy = []
                

                # PRE-COMPUTE: Leave-one-out means for all trials and timepoints
                # Shape: (n_trl, n_elc, n_time)
                loocv_means = np.zeros((n_trl, n_elc, n_time))
                
                total_sum = train_data.sum(axis=0)  # Sum over all trials: (n_elc, n_time)
                
                for trial in range(n_trl):
                    # Mean excluding this trial = (total_sum - this_trial) / (n_trl - 1)
                    loocv_means[trial] = (total_sum - train_data[trial]) / (n_trl - 1)
                
                #mat_entry = n_time * n_time
                outputs = np.asarray(Parallel(n_jobs=-1, backend='loky', verbose=51)(
                    delayed(process_timepoint)(ti, tj, train_data, test_data, loocv_means, test_stims, weights, neuron_prefs) for ti in range(n_time) for tj in range(n_time)
                )).reshape(n_time, n_time)
                # # Now do the temporal generalization
                # for ti in range(n_time):
                #     logging.basicConfig(level=logging.INFO)
                #     logger = logging.getLogger(__name__)
                #     logger.info(f'Processing {ti} of {n_time} timepoints')
                    
                #     for tj in range(n_time):
                #         outputs[ti, tj] = process_timepoint(ti, tj, train_data, test_data, loocv_means, test_stims, weights, neuron_prefs)
                    # outputs[ti, :] = np.array(
                    #     Parallel(n_jobs=-1, backend='loky')(
                    #         delayed(process_timepoint)(ti, tj, train_data, test_data, loocv_means, test_stims, weights, neuron_prefs) for tj in range(n_time)
                    #     )
                    # )
                    
                        # accuracy = []
                        # for trial in range(n_trl):
                        #     # Use pre-computed mean for this trial at training time ti
                        #     trn_mean_ti = loocv_means[trial, :, ti]
                            
                        #     # Demean the test data using this fold's training mean
                        #     test_data_demeaned = test_data[trial, :, tj] - trn_mean_ti
                            
                        #     # Apply spatial filter
                        #     v = weights[trial, :, :, ti]
                        #     pred = v @ test_data_demeaned
                            
                        #     # Decode
                        #     theta = np.inner(pred, np.exp(neuron_prefs * 1j))
                        #     decoded_angle = np.angle(theta)
                        #     decoded_error = np.abs(utils.circular_difference(decoded_angle, test_stims[trial]))
                        #     accuracy.append(np.cos(decoded_error))
                        
                        # outputs[ti, tj] = np.mean(accuracy)

            else: # test set cases, easier to do with vectorised calculations since the weights matrix is consistent for every trial
                weights = np.load(epoch_dir /  # chans x electrodes x timepoints
                                  'FE' / 
                                  'train_sk2_unif' / 
                                  f'{sub}_task-TG-MotEst-2.0_t-{t_min}-{t_max}_bl-{bl_min}-{bl_max}-{bpf_low}-{bpf_high}bpf-clean_FE-weights.npy')

                for ti in range(n_time): # for each training time
                    # logging
                    logging.basicConfig(level=logging.INFO)
                    logger = logging.getLogger(__name__)
                    logger.info(f'Processing {ti} of {n_time} timepoints')

                    v = weights[..., ti]

                    # Compute mean from ALL training data
                    trn_mean_ti = train_data[..., ti].mean(axis=0)  # Mean at training time ti
                    
                    for tj in range(n_time):
                        # Demean test data using training mean
                        test_data_demeaned = test_data[..., tj]# - trn_mean_ti
                        
                        v = weights[..., ti]
                        pred = (v @ test_data_demeaned.T).T

                        # decode the presented motion direction
                        theta = np.inner(pred, np.exp(neuron_prefs * 1j)) # Compute the complex vector sum of the channel responses
                        decoded_angle = np.angle(theta)
                        decoded_error = np.abs(utils.circular_difference(decoded_angle, test_stims))
                        accuracy = np.cos(decoded_error) # Decode the error in the presented motion direction

                        outputs[ti,tj] = np.mean(accuracy)

            output_dir = epoch_dir / 'FE' / 'train_sk2_unif' / 'tempGenMats'

            if not output_dir.exists():
                output_dir.mkdir(parents=True,exist_ok= True)

            #Save the outputs
            file_label = (
                f't-{t_min}-{t_max}_bl-{bl_min}-{bl_max}-{bpf_low}-{bpf_high}'
                'bpf-clean'
            )
            np.save(
                output_dir
                / f'trial-avgd-tempgen_matrix_trn-pk0_tst-pk{tst_pk[pk]}sk{tst_sk[sk]}_{file_label}.npy',
                outputs,
            )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run temporal generalisation for one or all participants."
    )
    parser.add_argument(
        "--subject",
        help="Process one participant directory, for example sub-001 (default: all participants).",
    )
    args = parser.parse_args(argv)

    for sub_dir in subject_directories(args.subject):
        print(f'Processing {sub_dir.name}')
        run_subject(sub_dir)


if __name__ == "__main__":
    main()

# %%

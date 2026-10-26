import os 
import pathlib 
import pandas as pd

def load_beh_data(rawdata_path: str, bads = False) -> pd.DataFrame:
    """
    Load behavioral data from the specified directory and filter it.

    Parameters:
    - rawdata_path (str): Path to the raw data directory.
    - bads (bool): If True, returns a summary of bad trials.

    Returns:
    - pd.DataFrame: Processed DataFrame containing the loaded and filtered data.
    - pd.DataFrame: Summary of bad trials if `bads` is True.
    """
    
    # Set working directory
    #os.chdir(r'C:\Users\uqtgastr\OneDrive - The University of Queensland\PhD\TG-MotEst-01\Scripts\BHV')
    
    # Load data
    rawdata = pathlib.Path(rawdata_path)
    
    li = {}
    
    for pt in rawdata.iterdir():

        if pt.is_dir():
            files = [entry for entry in (pt / "beh").iterdir() if "beh" in entry.name] # Just grab beh in case there's anything else there
            for file in files:
                if file.exists():
                    li[file.stem] = pd.read_csv(file, sep= '\t')
                else: 
                    raise ValueError(f"File {file} does not exist!")
        else:
            raise ValueError(f"Path {pt} is not a directory!")
            
    df = pd.concat(li, axis=0, ignore_index=True)
    
    # Drop unnecessary columns
    df = df.drop(['responseError'], axis=1)


    if 'session' in df.columns:
        df['trialInBlock'] = df.groupby(['subjectNumber', 'blockNr', 'session']).cumcount() + 1
    else:
        df['trialInBlock'] = df.groupby(['subjectNumber', 'blockNr']).cumcount() + 1

    bad_trial_summary = df.groupby('subjectNumber').agg({
    "responseTime": lambda x: (x < 0.2).sum(),
    "abort": lambda x: (x == "YES").sum()
    }).reset_index()
   
    # Filter data
    df = df.query("abort == 'NO' & responseTime > 0.2")
    
    df.loc[:,'priorKappa'] = round(df['priorKappa'] * 2) /2
    df.loc[:,'stimKappa'] = df['stimKappa'].round(3)
    df.loc[:,'priorKappa'] = df['priorKappa'].round(1)




    if bads:
         return df, bad_trial_summary
    else:
        return df


def load_decoding_data(derivatives_pth, file_label, cols, subdirectory='train_sk2_unif'):
    all = {}
    sorted_pts = sorted(derivatives_pth.iterdir(), key=lambda x: x.name.lower())
    for sub_dir in sorted_pts:
        if not (sub_dir.stem.startswith('sub') and sub_dir.is_dir()):
            continue    

        sub = sub_dir.stem


        metrics_dir = sub_dir / 'eeg' / 'FE' / subdirectory / 'decoding_metrics'

        fname = f'{sub}_task-TG-MotEst-2.0_{file_label}_FE-metrics-long.parquet'
        if not (metrics_dir / fname).exists():
            print(f'File not found: {metrics_dir / fname}')
            continue
        print(f'Loading parquet for {sub}/{len(sorted_pts)}...')
        all[sub] = pd.read_parquet(metrics_dir / fname, columns=cols)


    print('Putting them all together....')
    return pd.concat(all, axis=0, ignore_index=True)
    
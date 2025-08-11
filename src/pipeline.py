import mne
import numpy as np
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.channels import make_standard_montage
from mne import Epochs, events_from_annotations

GLOBAL_LABELS = {
    "rest": 0,
    "left": 1,
    "right": 2,
    "fists": 3,
    "feet": 4
}

RUNS_LEFT_RIGHT = [3, 4, 7, 8, 11, 12]
RUNS_FISTS_FEET = [5, 6, 9, 10, 13, 14]

def label_map(run, event_code):
    if event_code == 0:
        return GLOBAL_LABELS["rest"]
    elif event_code == 1:
        return GLOBAL_LABELS["left"] if run in RUNS_LEFT_RIGHT else GLOBAL_LABELS["fists"]
    elif event_code == 2:
        return GLOBAL_LABELS["right"] if run in RUNS_LEFT_RIGHT else GLOBAL_LABELS["feet"]
    else:
        return -1  # Undefined Run

class Pipeline:
    def __init__(self, subjects=45, runs : list[int]=[3, 4, 7, 8, 11, 12], fmin : float = 8.0, fmax : float = 25, tmin : float = 0.0, tmax : float = 4.0):
        if subjects is None:
            subjects = list(range(1, 110)) # Load all subjectss
        if runs is None:
            runs = RUNS_LEFT_RIGHT + RUNS_FISTS_FEET # Load All the data from db

        self.subjects = subjects
        self.runs = runs
        self.fmin = fmin
        self.fmax = fmax
        self.tmin = tmin
        self.tmax = tmax
        self.X = []
        self.Y = []

        self.load_all_data()

    def load_all_data(self):
        for subject in self.subjects:
            for run in self.runs:
                try:
                    file_names = eegbci.load_data(subject, [run])
                    raws = [read_raw_edf(f, preload=True) for f in file_names]
                    raw = concatenate_raws(raws)
                    eegbci.standardize(raw)
                    montage = make_standard_montage("standard_1005") # This is the closest standard to standard_1010 system..
                    raw.set_montage(montage)
                    raw.filter(self.fmin, self.fmax)
                    
                    # These are the default, will change them to my customs
                    event_id = {"T0": 0, "T1": 1, "T2": 2}
                    events, _ = events_from_annotations(raw, event_id=event_id)
                    if len(events) == 0:
                        print(f"No events found for subject {subject} run {run}")
                        continue

                    epochs = Epochs(raw, events, event_id, tmin=self.tmin, tmax=self.tmax, baseline=None, preload=True)
                    X_run = epochs.get_data()
                    # Map event codes to the customs ones...
                    Y_run = np.array([label_map(run, e[-1]) for e in epochs.events])

                    self.X.append(X_run)
                    self.Y.append(Y_run)
                except Exception as e:
                    print(f"having problems loading subject {subject} run {run}: {e}")

        if self.X and self.Y:
            self.X = np.concatenate(self.X, axis=0)
            self.Y = np.concatenate(self.Y, axis=0)
        else:
            self.X = np.array([])
            self.Y = np.array([])

# Usage of the classs : Example  
#pipeline = Pipeline(subjects=[1], runs=[1, 3, 5, 7, 8, 10])
#print(pipeline.Y)
#print(pipeline.X.shape)
#print(pipeline.Y.shape)

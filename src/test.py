#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Samukel
# Description: EEG Motor Imagery Classification Pipeline
import mne
import numpy as np
import joblib
import matplotlib.pyplot as plt

from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.channels import make_standard_montage
from mne import Epochs, events_from_annotations
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from matplotlib.animation import FuncAnimation
from collections import deque


def load_and_preprocess(subject: int = 45, runs: list[int] = [3, 4, 7, 8, 11, 12], fmin=8, fmax=25):
    """
    Load and preprocess EEG data from BCI Competition IV dataset.

    Parameters:
    - subject: int, subject ID (1–109)
    - runs: list of run numbers to include
    - fmin, fmax: frequency band for filtering

    Returns:
    - raw: preprocessed MNE Raw object
    """
    allowed_runs = {3, 4, 7, 8, 11, 12}
    if not set(runs).issubset(allowed_runs):
        raise ValueError(f"Runs must be a subset of {allowed_runs}. Provided: {runs}")
    if subject not in range(1, 110):
        raise ValueError("Subject ID must be between 1 and 109.")

    raw_fnames = eegbci.load_data(subject, runs)
    raws = [read_raw_edf(f, preload=True) for f in raw_fnames]
    raw = concatenate_raws(raws)

    eegbci.standardize(raw)
    montage = make_standard_montage("standard_1005")
    raw.set_montage(montage)

    raw.filter(fmin, fmax)
    return raw


def extract_epochs(raw):
    """
    Extract epochs and labels from raw EEG using event annotations.

    Returns:
    - X: np.array, epochs data (n_epochs, n_channels, n_times)
    - y: np.array, corresponding labels
    """
    event_id = dict(left=0, right=1)
    events, _ = events_from_annotations(raw, event_id=dict(T1=0, T2=1))
    epochs = Epochs(raw, events, event_id, tmin=0, tmax=4.0, baseline=None)
    X = epochs.get_data()
    y = epochs.events[:, -1]
    return X, y


def train_csp_lda(X, y, n_csp=8):
    """
    Train a CSP + LDA model.

    Returns:
    - csp: trained CSP object
    - lda: trained LDA object
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    csp = CSP(n_components=n_csp)
    lda = LDA()

    csp.fit(X_train, y_train)
    lda.fit(csp.transform(X_train), y_train)

    y_pred = lda.predict(csp.transform(X_test))
    auroc = roc_auc_score(y_test, y_pred)
    print(f"[INFO] Test AUROC: {auroc:.3f}")
    return csp, lda


def save_models(csp, lda, csp_path="../assets/Models/csp_model.joblib", lda_path="../assets/Models/lda_model.joblib"):
    joblib.dump(csp, csp_path)
    joblib.dump(lda, lda_path)


def load_models(csp_path="../assets/Models/csp_model.joblib", lda_path="../assets/Models/lda_model.joblib"):
    csp = joblib.load(csp_path)
    lda = joblib.load(lda_path)
    return csp, lda


def predict_new(raw, csp, lda):
    X_new, _ = extract_epochs(raw)
    features_new = csp.transform(X_new)
    predictions = lda.predict(features_new)
    return predictions


def stream_plot_with_psd(X_stream, y_stream, csp_loaded, lda_loaded, sfreq, window=5):
    preds = deque(maxlen=window)
    trues = deque(maxlen=window)

    fig, (ax_pred, ax_psd) = plt.subplots(1, 2, figsize=(12, 5))

    ax_pred.set_ylim(-0.5, 2.5)
    ax_pred.set_xlim(0, window - 1)
    ax_pred.set_title('Streaming Prediction (Blue=True, Red=Predicted)')
    ax_pred.set_xlabel('Epoch')
    ax_pred.set_ylabel('Label')

    true_plot, = ax_pred.plot([], [], 'bo-', label='True Label')
    pred_plot, = ax_pred.plot([], [], 'ro-', label='Predicted Label')
    ax_pred.legend()

    def update(frame):
        epoch = X_stream[frame:frame + 1]
        true_label = y_stream[frame]
        pred = lda_loaded.predict(csp_loaded.transform(epoch))[0]

        trues.append(true_label)
        preds.append(pred)
        x_vals = list(range(max(0, frame - window + 1), frame + 1))

        true_plot.set_data(x_vals, list(trues))
        pred_plot.set_data(x_vals, list(preds))
        ax_pred.set_xlim(x_vals[0], x_vals[-1])

        # PSD
        psd, freqs = mne.time_frequency.psd_array_welch(
            epoch[0], sfreq=sfreq, fmin=0.1, fmax=40, n_fft=256, verbose=False
        )

        ax_psd.clear()
        ax_psd.plot(freqs, 10 * np.log10(psd.T))
        ax_psd.set_xlim(0, 40)
        ax_psd.set_xlabel('Frequency (Hz)')
        ax_psd.set_ylabel('PSD (dB/Hz)')
        ax_psd.set_title(f'PSD of Epoch {frame + 1}')

        return true_plot, pred_plot, ax_psd

    ani = FuncAnimation(fig, update, frames=len(X_stream), interval=500, blit=True, repeat=False)
    plt.tight_layout()
    plt.show()
    return ani


if __name__ == "__main__":
    # Train on subject 45
    raw = load_and_preprocess(subject=45)
    X, y = extract_epochs(raw)
    csp, lda = train_csp_lda(X, y, n_csp=2)
    save_models(csp, lda)

    # Evaluate on unseen subject 30
    raw_unseen = load_and_preprocess(subject=30)
    X_unseen, y_unseen = extract_epochs(raw_unseen)

    if len(y_unseen) < 10:
        print("Warning: metrics may be unreliable.")

    preds_unseen = predict_new(raw_unseen, csp, lda)

    accuracy = np.mean(preds_unseen == y_unseen)
    auroc = roc_auc_score(y_unseen, preds_unseen)

    print(f"Unseen Accuracy: {accuracy:.2f}")
    print(f"Unseen AUROC: {auroc:.2f}")

    cm = confusion_matrix(y_unseen, preds_unseen)
    print("Confusion Matrix:\n", cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Left (0)', 'Right (1)'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (Unseen Data)")
    plt.show()

    # will be needed later on, also has to e fixed.. stream visualization
    # stream_plot_with_psd(X_unseen, y_unseen, csp, lda, raw_unseen.info['sfreq'])

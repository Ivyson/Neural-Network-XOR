# Non Invasive Electro-Encephalography Pipeline
Its 2025/07/22, I am close to my death bed than i have been the past 2 months. This project is going to kill me.. If this pipeline does not work out... I am more than free to free my soul from this shitty world!

# EEG Motor Imagery Model Pipeline — Detailed TODO List

## 1. File Preparation  
- [ ] Download all EDF files for all sessions and subjects from [PhysioNet EEGMMIDB](https://physionet.org/content/eegmmidb/1.0.0/).
- [ ] Organize files into a directory structure:
  - Example: `data/raw/{subject_id}/{session_id}.edf`
- [ ] Ensure that the files are not corrupted, Ensure that they downloaded successfully etc..
- [ ] Create a mapping of runs/sessions to task types for correct label assignment (see dataset documentation).
    - For an example
      - T0 corresponds to rest
      - T1 corresponds to onset of motion (real or imagined) of
the left fist (in runs 3, 4, 7, 8, 11, and 12)
both fists (in runs 5, 6, 9, 10, 13, and 14)
      - T2 corresponds to onset of motion (real or imagined) of
the right fist (in runs 3, 4, 7, 8, 11, and 12)
both feet (in runs 5, 6, 9, 10, 13, and 14)

## 2. Data Inspection & Quality Control  
- [ ] For each file:
  - [ ] Confirm the presence of annotations (`T0`, `T1`, `T2`).
  - [ ] Visualize a handful of EEG traces to spot major artifacts or flat channels.
  - [ ] Log or flag any files with missing channels or clear corruption for exclusion.

## 3. Preprocessing  
- [ ] For each EDF file, perform:
  - [ ] Load the file using MNE-Python or equivalent.
  - [ ] Apply a bandpass filter (e.g., 0.5–40 Hz).
  - [ ] (Optional) Detect and remove bad channels (manual or automated).
  - [ ] (Optional, advanced) Perform artifact removal (e.g., ICA for blinks/EOG).
  - [ ] Save the preprocessed data (in-memory or to disk if needed).

## 4. Event Annotation & Epoch Extraction  
- [ ] Parse annotations for each file:
  - [ ] Extract all T0, T1, and T2 events.
  - [ ] Map each annotation to its correct class label based on the run/session.
- [ ] For each event:
  - [ ] Epoch the data (e.g., from event onset to 2 seconds after).
  - [ ] Discard epochs with too much noise/artifact (if possible).
  - [ ] Assign the correct label (rest, left, right, both, feet) to each epoch.
  - [ ] Save each epoch and its label to disk (e.g., as `.npy`, `.h5`, or similar).

## 5. Data Aggregation  
- [ ] Aggregate all epochs and labels from all files to create the full dataset.
  - [ ] Ensure data format is consistent (e.g., shape: samples × channels × time).
  - [ ] Store the dataset and a matching label vector.

## 6. Dataset Splitting  
- [ ] Split the data into training, validation, and test sets.
  - [ ] Prefer splitting by subject or session for better generalization.
  - [ ] Ensure class balance in each split.
  - [ ] Save split indices/lists for reproducibility.

## 7. Model Architecture & Data Loading  
- [ ] Define the deep learning model (CNN, LSTM, or hybrid).
- [ ] Implement a custom data loader to batch epochs and labels efficiently.
- [ ] Set up input normalization/standardization logic (if needed).

## 8. Model Training  
- [ ] Set up training loop, loss function (cross-entropy), optimizer, and callbacks.
- [ ] Train the model on the training set, validating on the validation set.
- [ ] Log loss/accuracy and save the best-performing model checkpoint.
- [ ] Apply early stopping or regularization as needed.

## 9. Model Evaluation  
- [ ] Evaluate the final model on the test set.
  - [ ] Report accuracy, confusion matrix, and per-class metrics.
  - [ ] Optionally, evaluate generalization by subject/session.

## 10. Hyperparameter Tuning & Experimentation  
- [ ] Repeat training with different:
  - [ ] Epoch lengths, filter settings, and preprocessing steps.
  - [ ] Model architectures (CNN-only, LSTM-only, CNN-LSTM, etc.).
  - [ ] Regularization and augmentation strategies.
- [ ] Record results for each experiment.

## 11. Interpretation & Reporting  
- [ ] Visualize learned model filters, activation maps, or saliency (if feasible).
- [ ] Analyze which classes/tasks are most/least difficult for the model.
- [ ] Document all pipeline steps, parameters, and reproducibility details.

## 12. Packaging & Automation  
- [ ] Bundle preprocessing, training, and evaluation code as scripts or notebooks.
- [ ] Write a README explaining how to reproduce the full pipeline.
- [ ] (Optional) Containerize or provide environment files (e.g., `requirements.txt`, `environment.yml`).

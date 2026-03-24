# import libraries
import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])

import os
import pandas as pd
import collections
from sklearn.model_selection import StratifiedKFold, train_test_split

import src.utils.util_general as util_general

# seed everything
seed = 0
util_general.seed_all(seed)

# cross-validation class
class Cross_Validation:
    """
    A class to handle cross-validation on a given dataset.

    Attributes:
    ----------
    dataset : str
        The dataset to be used for cross-validation (CLARO, radiomics).
    data_file : str
        The file path to the dataset file.
    dest_dir : str
        The directory where the processed data should be stored.
    dest_dir_folds : str
        The directory where the cross-validation folds should be stored.
    split_col : str
        The column name based on which the dataset will be split.
    surv_time : str
        The column name representing survival time.
    y_label : str
        The target variable column name.
    cv : int
        The number of cross-validation folds.
    val_size : float
        The size of the validation set as a proportion of the dataset.
    """

    def __init__(self, dataset, data_file, dest_dir, dest_dir_folds, split_col, surv_time, y_label, cv, val_size):
        # initialize attributes
        self.dataset = dataset
        self.data_file = data_file
        self.dest_dir = dest_dir
        self.dest_dir_folds = dest_dir_folds
        self.split_col = split_col
        self.surv_time = surv_time
        self.y_label = y_label
        self.cv = cv
        self.val_size = val_size
    
    def create_cv_files(self):
        # load data
        data = pd.read_csv(self.data_file)

        # handle missing patients
        if self.dataset == 'radiomics':
            missing_patients = ['LUNG1-014', 'LUNG1-021', 'LUNG1-085', 'LUNG1-095', 'LUNG1-128', 'LUNG1-194', 'LUNG1-246']
        elif self.dataset == 'radgenomics':
            missing_patients = ['AMC-001', 'AMC-002', 'AMC-003', 'AMC-004', 'AMC-005', 'AMC-006', 'AMC-007', 'AMC-008', 'AMC-009', 'AMC-010', 'AMC-011', 'AMC-012', 'AMC-013', 'AMC-014', 'AMC-015', 'AMC-016', 'AMC-017', 'AMC-018', 'AMC-019', 'AMC-020', 'AMC-021', 'AMC-022', 'AMC-023', 'AMC-024', 'AMC-025', 'AMC-026', 'AMC-027', 'AMC-028', 'AMC-029', 'AMC-030', 'AMC-031', 'AMC-032', 'AMC-033', 'AMC-034', 'AMC-035', 'AMC-036', 'AMC-037', 'AMC-038', 'AMC-039', 'AMC-040', 'AMC-041', 'AMC-042', 'AMC-043', 'AMC-044', 'AMC-045', 'AMC-046', 'AMC-047', 'AMC-048', 'AMC-049', 'R01-009', 'R01-143', 'R01-147', 'R01-148', 'R01-149', 'R01-150', 'R01-151', 'R01-152', 'R01-153', 'R01-154', 'R01-155', 'R01-156', 'R01-158', 'R01-159', 'R01-160', 'R01-161', 'R01-162', 'R01-163']
        else:
            missing_patients = []
        data = data[~data[self.split_col].isin(missing_patients)]
        data_split = data[[self.split_col, self.surv_time, self.y_label]].reset_index(drop=True).drop_duplicates()

        # convert survival time to months
        data_split[surv_time] = data_split[surv_time] // 30

        # convert survival time to binary label in 2 years
        data_split[y_label] = data_split.apply(lambda row: 0 if row[surv_time] > 24 else row[y_label], axis=1)

        # cut survival times greater than 2 years
        data_split[surv_time] = data_split.apply(lambda row: 24 if row[surv_time] > 24 else row[surv_time], axis=1)

        # k-folds cv
        fold_data = collections.defaultdict(lambda: {})
        skf = StratifiedKFold(n_splits=self.cv, random_state=42, shuffle=True)

        for fold, (train_index, test_index) in enumerate(skf.split(data_split, data_split[self.y_label])):
            train, test = data_split.iloc[train_index], data_split.iloc[test_index]
            train, val = train_test_split(train, test_size=self.val_size, stratify=train[self.y_label], random_state=42)

            fold_data[fold]['train'] = train[self.split_col].tolist()
            fold_data[fold]['val'] = val[self.split_col].tolist()
            fold_data[fold]['test'] = test[self.split_col].tolist()

        # all.csv
        with open(os.path.join(self.dest_dir, 'all.csv'), 'w', newline='') as file:
            # generate the list of dictionaries for the DataFrame rows
            rows = []
            for patient in data_split[self.split_col]:
                label = "%s" % int(data_split.loc[data_split[self.split_col] == patient][self.y_label].item())
                time = "%s" % data_split.loc[data_split[self.split_col] == patient][self.surv_time].item()
                row = {'patientID': str(patient), 'time': time, 'label': label}
                rows.append(row)
            
            # generate the DataFrame from the list of dictionaries
            all = pd.DataFrame(rows)
            all = all.rename(columns={'patientID': 'PatientID', 'time': 'Time', 'label': 'Label'})
            
            # save the DataFrame as a CSV file
            all.to_csv(file, index=False, sep=',')

        # create split dir
        steps = ['train', 'val', 'test']
        for fold in range(cv):
            dest_dir_cv = os.path.join(self.dest_dir_folds, str(fold))
            util_general.create_dir(dest_dir_cv)

            # .csv
            for step in steps:
                with open(os.path.join(dest_dir_cv, '%s.csv' % step), 'w', newline='') as file:
                    # generate the list of dictionaries for the DataFrame rows
                    rows = []
                    for patient in fold_data[fold][step]:
                        label = "%s" % data_split.loc[data_split[self.split_col] == patient, self.y_label].item()
                        time = "%s" % data_split.loc[data_split[self.split_col] == patient][self.surv_time].item()
                        row = {'patientID': str(patient), 'time': time, 'label': label}
                        rows.append(row)
                
                    # generate the DataFrame from the list of dictionaries
                    df = pd.DataFrame(rows)
                    df = df.rename(columns={'patientID': 'PatientID', 'time': 'Time', 'label': 'Label'})
                    
                    # save the DataFrame as a CSV file
                    df.to_csv(file, index=False, sep=',')


if __name__=="__main__":

    # params
    dataset   = 'radiomics'
    data_file = "data/radiomics/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv"
    y_label   = "deadstatus.event"
    surv_time = "Survival.time"
    split_col = "PatientID"
    cv        = 10
    val_size  = 0.1

    # files and directories
    dest_dir = "./data/processed/radiomics/os_2y/folds"
    dest_dir_folds = os.path.join(dest_dir, str(cv))
    util_general.create_dir(dest_dir)
    
    # cv
    cross_val = Cross_Validation(data_file=data_file, dest_dir=dest_dir, dest_dir_folds=dest_dir_folds, 
                                split_col=split_col, surv_time=surv_time, y_label=y_label, dataset=dataset,
                                cv=cv, val_size=val_size)
    cross_val.create_cv_files()

    
# Don's support time-window

import os
import argparse
import numpy as np
import pandas as pd
from scipy import sparse
from collections import defaultdict
from sklearn.preprocessing import OneHotEncoder
import time

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def phi(x):
    return np.log(1 + x)

phi_vectorized = np.vectorize(phi)


WINDOW_LENGTHS = [3600 * 24 * 30, 3600 * 24 * 7, 3600 * 24, 3600]
NUM_WINDOWS = len(WINDOW_LENGTHS) + 1

def single_to_sparse(df, Q_mat, user_id,
                     item_id, item_count, item_success, 
                     skills, skill_counts, skill_successes, 
                     total_count, total_success, active_features):
    features = {}
    num_items, num_skills = Q_mat.shape

    # Keep track of original dataset
    features['df'] = np.empty((0, len(df.keys())))

    # Skill features
    if 's' in active_features:
        features["s"] = sparse.csr_matrix(np.empty((0, num_skills)))

    # Past attempts and wins features
    for key in ['a', 'w']:
        if key in active_features:
            if 'tw' in active_features:
                features[key] = sparse.csr_matrix(np.empty((0, (num_skills + 2) * NUM_WINDOWS)))
            else:
                features[key] = sparse.csr_matrix(np.empty((0, num_skills + 2)))

    # Current skills one hot encoding
    if 's' in active_features:
        features['s'] = sparse.csr_matrix(skills)

    # Attempts
    if 'a' in active_features:
        # Counts
        attempts = np.zeros((1, num_skills + 2))

        # Past attempts for relevant skills
        if 'sc' in active_features:
            attempts[:, :num_skills] = phi(skill_counts*skills)

        # Past attempts for item
        if 'ic' in active_features:
            attempts[:, -2] = phi(np.array([item_count]))

        # Past attempts for all items
        if 'tc' in active_features:
            attempts[:, -1] = phi(np.array([total_count]))

        features['a'] = sparse.csr_matrix(attempts)

    # Wins
    if "w" in active_features:
        wins = np.zeros((1, num_skills + 2))

        # Past wins for relevant skills
        if 'sc' in active_features:
            wins[:, :num_skills] = phi(skill_successes*skills)

        # Past wins for item
        if 'ic' in active_features:
            wins[:, -2] = phi(np.array([item_success]))

        # Past wins for all items
        if 'tc' in active_features:
            wins[:, -1] = phi(np.array([total_success]))

        features['w'] = sparse.csr_matrix(wins)

    # User and item one hot encodings
    if 'u' in active_features:
        features['u'] = sparse.csr_matrix((
            [1], ([0], [user_id])
        ), shape=(1, num_items))
    if 'i' in active_features:
        features['i'] = sparse.csr_matrix((
            [1], ([0], [item_id])
        ), shape=(1, num_items))

    X = sparse.hstack([features[x] for x in features.keys() if x != 'df']).tocsr()
    return X


def df_to_sparse(df, Q_mat, active_features):
    """Build sparse dataset from dense dataset and q-matrix.

    Arguments:
        df (pandas DataFrame): output by prepare_data.py
        Q_mat (sparse array): q-matrix, output by prepare_data.py
        active_features (list of str): features

    Output:
        sparse_df (sparse array): sparse dataset where first 5 columns are the same as in df
    """
    features = {}
    num_items, num_skills = Q_mat.shape

    # Keep track of original dataset
    features['df'] = np.empty((0, len(df.keys())))

    # Skill features
    if 's' in active_features:
        features["s"] = sparse.csr_matrix(np.empty((0, num_skills)))

    # Past attempts and wins features
    for key in ['a', 'w']:
        if key in active_features:
            if 'tw' in active_features:
                features[key] = sparse.csr_matrix(np.empty((0, (num_skills + 2) * NUM_WINDOWS)))
            else:
                features[key] = sparse.csr_matrix(np.empty((0, num_skills + 2)))

    # Build feature rows by iterating over users
    for user_id in df["user_id"].unique():
        # print(f'run - {user_id}')
        # df_user = df[df["user_id"] == user_id][["user_id", "item_id", "timestamp", "correct", "skill_id", "opp"]].copy()
        df_user = df[df["user_id"] == user_id][["user_id", "item_id", "timestamp", "correct", "skill_id"]].copy()
        df_user = df_user.values
        num_items_user = df_user.shape[0]

        skills = Q_mat[df_user[:, 1].astype(int)].copy()

        features['df'] = np.vstack((features['df'], df_user))

        item_ids = df_user[:, 1].reshape(-1, 1)
        item_ids_flat = item_ids.flatten()
        labels = df_user[:, 3].reshape(-1, 1)
        labels_flat = labels.flatten()

        # Current skills one hot encoding
        if 's' in active_features:
            features['s'] = sparse.vstack([features["s"], sparse.csr_matrix(skills)])

        # Attempts
        if 'a' in active_features:
            # Counts
            attempts = np.zeros((num_items_user, num_skills + 2))

            # Past attempts for relevant skills
            if 'sc' in active_features:
                tmp = np.vstack((np.zeros(num_skills), skills))[:-1]
                attempts[:, :num_skills] = phi(np.cumsum(tmp, 0) * skills)

            # Past attempts for item
            if 'ic' in active_features:
                item_count = {i: 0 for i in item_ids_flat}
                ic_attempts = []
                for i in item_ids_flat:
                    ic_attempts.append(item_count[i])
                    item_count[i] += 1
                attempts[:, -2] = phi(np.array(ic_attempts))

            # Past attempts for all items
            if 'tc' in active_features:
                attempts[:, -1] = phi(np.arange(num_items_user))

            features['a'] = sparse.vstack([features['a'], sparse.csr_matrix(attempts)])

        # Wins
        if "w" in active_features:
            wins = np.zeros((num_items_user, num_skills + 2))

            # Past wins for relevant skills
            if 'sc' in active_features:
                tmp = np.vstack((np.zeros(num_skills), skills))[:-1]
                corrects = np.hstack((np.array(0), df_user[:, 3])).reshape(-1, 1)[:-1]
                wins[:, :num_skills] = phi(np.cumsum(tmp * corrects, 0) * skills)

            # Past wins for item
            if 'ic' in active_features:
                item_win_count = {i: 0 for i in item_ids_flat}
                ic_wins = []
                for idx, i in enumerate(item_ids_flat):
                    ic_wins.append(item_win_count[i])
                    item_win_count[i] += labels_flat[idx]
                wins[:, -2] = phi(np.array(ic_wins))

            # Past wins for all items
            if 'tc' in active_features:
                wins[:, -1] = phi(np.concatenate((np.zeros(1), np.cumsum(df_user[:, 3])[:-1])))

            features['w'] = sparse.vstack([features['w'], sparse.csr_matrix(wins)])

        # break

    # User and item one hot encodings
    l = len(features["df"])
    if 'i' in active_features:
        features['i'] = sparse.csr_matrix((
            np.ones(l), (np.arange(l), features["df"][:, 1].reshape(-1))
        ), shape=(l, num_items))

    X = sparse.hstack([sparse.csr_matrix(features['df']),
                       sparse.hstack([features[x] for x in features.keys() if x != 'df'])]).tocsr()

    return X


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Encode sparse feature matrix for logistic regression.')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('-sim', action='store_true')
    parser.add_argument('-u', action='store_true',
                        help='If True, include user one hot encoding.')
    parser.add_argument('-i', action='store_true',
                        help='If True, include item one hot encoding.')
    parser.add_argument('-s', action='store_true',
                        help='If True, include skills many hot encoding .')
    parser.add_argument('-ic', action='store_true',
                        help='If True, include item historical counts.')
    parser.add_argument('-sc', action='store_true',
                        help='If True, include skills historical counts.')
    parser.add_argument('-tc', action='store_true',
                        help='If True, include total historical counts.')
    parser.add_argument('-w', action='store_true',
                        help='If True, historical counts include wins.')
    parser.add_argument('-a', action='store_true',
                        help='If True, historical counts include attempts.')
    parser.add_argument('-tw', action='store_true',
                        help='If True, historical counts are encoded as time windows.')
    args = parser.parse_args()


    print(f"encoding: {args.dataset}")
    #_dir = "simulation/simulated-data" if args.sim else "data"
    _file = f"{args.dataset}.csv" if args.sim else "preprocessed_data.csv"

    orig_data_path = os.path.join("../../data/real", args.dataset)
    data_path = args.dataset
    df = pd.read_csv(os.path.join(data_path, _file), sep="\t")
    df = df[["user_id", "item_id", "timestamp", "correct", "skill_id"]]
    Q_mat = sparse.load_npz(os.path.join(orig_data_path, 'q_mat.npz')).toarray()

    all_features = ['u', 'i', 's', 'ic', 'sc', 'tc', 'w', 'a', 'tw']
    active_features = [features for features in all_features if vars(args)[features]]
    features_suffix = ''.join(active_features)

    X = df_to_sparse(df, Q_mat, active_features)
    sparse.save_npz(os.path.join(data_path, f"X-{features_suffix}"), X)

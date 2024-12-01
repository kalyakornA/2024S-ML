import pandas as pd
import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.utils import resample
from tqdm import tqdm


def get_dataset(train_ratio, random_seed):
    file_path = os.getcwd() + "/exp3-reviews.csv"
    df = pd.read_csv(file_path, delimiter="\t")
    
    x = df['reviewText'].tolist()
    y = df['overall'].tolist()

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_ratio, random_state=random_seed)
    tfidf = TfidfVectorizer()
    x_train_tfidf = tfidf.fit_transform(x_train)
    x_test_tfidf = tfidf.transform(x_test)

    return x_train_tfidf, x_test_tfidf, y_train, y_test


class BaggingClassifier:
    def __init__(self, base_classifier, n_estimators=2, random_state=None):
        self.base_classifier = base_classifier
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.estimators = []

    def fit(self, X, y):
        for i in tqdm(range(self.n_estimators)):
            X_subset, y_subset = resample(X, y, replace=True, n_samples=len(y), random_state=self.random_state)
            classifier = self.base_classifier()
            classifier.fit(X_subset, y_subset)
            self.estimators.append(classifier)

    def predict(self, x):
        predictions = np.zeros((x.shape[0], self.n_estimators), dtype=int)
        for i, estimator in enumerate(self.estimators):
            predictions[:, i] = estimator.predict(x)
        return np.mean(predictions, axis=1)


class AdaBoostClassifier:
    def __init__(self, base_classifier, n_estimators=10):
        self.base_classifier = base_classifier
        self.n_estimators = n_estimators
        self.estimators = []
        self.alphas = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        weights = np.ones(n_samples) / n_samples

        for _ in tqdm(range(self.n_estimators)):
            # (a) Fit weak classifier and predict labels
            X_subset, y_subset = resample(X, y, replace=True, n_samples=n_samples)            
            classifier = self.base_classifier()
            classifier.fit(X_subset, y_subset, sample_weight=weights*n_samples)
            self.estimators.append(classifier)
            y_pred = classifier.predict(X_subset)

            # (b) Compute Error
            error = np.sum(weights * (y_pred != y))
            # (c) Compute Alpha
            alpha = 0.5 * np.log((1 - error) / max(error, 1e-10))
            self.alphas.append(alpha)
            # (d) Update weights
            weights *= np.exp(-alpha * np.array(y_subset) * y_pred)
            weights /= np.sum(weights)
               
    def predict(self, X):
        y_pred = []
        for regressor in self.estimators:
            y_pred.append(regressor.predict(X))
        y_pred = np.array(y_pred).T
  
        weighted_medians = []
        alphas_log = np.log(1 / np.array(self.alphas))
        for i in range(len(y_pred)):
            sorted_indices = sorted(range(len(y_pred[i])), key=lambda x: y_pred[i][x])
            cumulative_weight = 0
            median_idx = 0
            for idx in sorted_indices:
                cumulative_weight += alphas_log[idx]
                if cumulative_weight >= 0.5:
                    median_idx = idx
                    break
            weighted_medians.append(y_pred[i][median_idx])
        return np.array(weighted_medians)
        

def baseline(x, y, test):
    print("FINDING BASELINE")
    svc = LinearSVC()
    svc.fit(x, y)
    svc_pred = svc.predict(test)
    
    mae = mean_absolute_error(y_test, svc_pred)
    mse = mean_squared_error(y_test, svc_pred)
    rmse = np.sqrt(mean_squared_error(y_test, svc_pred))

    print(f"SVC_PRED=\nMAE: {mae}\nMSE: {mse}\nRMSE: {rmse}")

    tree = DecisionTreeClassifier()
    tree.fit(x, y)
    tree_pred = tree.predict(test)

    mae = mean_absolute_error(y_test, tree_pred)
    mse = mean_squared_error(y_test, tree_pred)
    rmse = np.sqrt(mean_squared_error(y_test, tree_pred))

    print(f"TREE_PRED=\nMAE: {mae}\nMSE: {mse}\nRMSE: {rmse}")
    return svc_pred, tree_pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ensemble', choices=['bagging', 'adaboost'])       ## bagging or boost
    parser.add_argument('--classifier', choices=['svm', 'decision_tree'])     ## svm or decision tree

    args = parser.parse_args()
    
    train_ratio = 0.9
    random_seed = 0
    n_estimators = 2
    random_state = 0
    x_train, x_test, y_train, y_test = get_dataset(train_ratio, random_seed)

    if not args.ensemble and not args.classifier:
        baseline(x_train, y_train, x_test)
        exit()

    if args.classifier == "svm":
        base_classifier = LinearSVC
    elif args.classifier == "decision_tree":
        base_classifier = DecisionTreeClassifier
    else:
        raise ValueError("Invalid Classifier")
    
    if args.ensemble == "bagging":
        ensemble = BaggingClassifier(base_classifier, n_estimators, random_state)
    elif args.ensemble == "adaboost":
        ensemble = AdaBoostClassifier(base_classifier, n_estimators)

    ensemble.fit(x_train, y_train)
    y_pred = ensemble.predict(x_test)
    print(y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"{args.ensemble} ensemble, {args.classifier} classifier\nMAE: {mae}\nMSE: {mse}\nRMSE: {rmse}")

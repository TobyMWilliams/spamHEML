import pickle
import numpy as np
import math
import os


def read_pickle(filepath):
    """
    Load a pickled object from disk.
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def compute_mcc(cm):
    """
    Compute Matthews Correlation Coefficient for a 2×2 confusion matrix.
    Assumes cm is in order [[TP, FP], [FN, TN]].
    """
    cm = np.asarray(cm)
    if cm.size != 4:
        return None

    tp, fp, fn, tn = cm.ravel()
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return ((tp * tn) - (fp * fn)) / denom if denom else 0


def compute_f1(cm):
    """
    Compute F1 score for a 2×2 confusion matrix.
    Assumes cm is in order [[TP, FP], [FN, TN]].
    """
    cm = np.asarray(cm)
    if cm.size != 4:
        return None

    tp, fp, fn, tn = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    return (2 * precision * recall) / (precision + recall) if (precision + recall) else 0



# method to cycle through the saved data and report the results
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    matrices_dir = os.path.join(script_dir, 'data', 'featureData', 'matrices')

    subdirs = ['45Features', '30Features', '60Features']
    features = ['GloVe', 'BoW', 'BIGRAM', 'TRIGRAM', 'TFIDF']
    reductions = ['chi2','pca' , 'ica' ,'svd',  'nmf' ] #      
    models = ['logregCM.pkl', 'nbCM.pkl', 'svmCM.pkl']

    # Store MCC and F1 results per model
    model_results = {model: [] for model in models}

    for red in reductions:
        for feat in features:
            for sd in subdirs:
                base_path = os.path.join(matrices_dir, sd, feat, red)
                for model_file in models:
                    path = os.path.join(base_path, model_file)
                    if not os.path.isfile(path):
                        print(f"Warning: File not found: {path}")
                        continue

                    cm = read_pickle(path)
                    mcc = compute_mcc(cm)
                    f1 = compute_f1(cm)

                    if mcc is None or f1 is None:
                        continue

                    # Track MCC and F1 for each model configuration
                    model_results[model_file].append((mcc, f1, feat, red, sd, model_file))

    # Flatten all results for global ranking
    combined_results = []
    for model_file, results in model_results.items():
        for mcc, f1, feat, red, sd, _ in results:
            config = f"{feat}-{red}-{sd}"
            combined_results.append((mcc, f1, config, model_file))

    # Sort globally by MCC
    combined_results_sorted = sorted(combined_results, key=lambda x: x[0], reverse=True)

    print("\n=== Global Rankings (All Models) ===\n")
    for rank, (mcc, f1, config, model_file) in enumerate(combined_results_sorted, 1):
        print(f"{rank:3d}. MCC: {mcc:.4f} | F1: {f1:.4f} | Config: {config} | Model: {model_file}")


if __name__ == '__main__':
    main()

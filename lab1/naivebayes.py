from collections import Counter
import json
import random
import math
import re
from tqdm import tqdm


# 1: Partitioning Training set and Testing set
def dataset(train_prob, random_seed=None):
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    ## Read label folder
    data = []
    with open('./trec06p/label/index', 'r') as f:
        data = f.readlines()

    ## Divide train set and test set
    random.shuffle(data)
    split_index = int(len(data) * train_prob)
    train_paths, test_paths = data[:split_index], data[split_index:]

    ## Write train set and test set to files
    with open('dataset/trainset', 'w') as f:
        f.writelines(train_paths)
    with open('dataset/testset', 'w') as f:
        f.writelines(test_paths)


# 2: Train Data, create y and x|y frequency
def train(sample_rate=1):
    ## P(Xi|Y) Word Frequency dictionary {ham: {word1: freq, ..}, spam: {word2: freq, ..}}
    word_freq = {'ham': Counter(), 'spam': Counter()}
    ## P(Y) Total Frequency dictionary {ham: freq, spam: freq}
    total_freq = {'ham': 0, 'spam': 0}

    ## Read each line in train set
    train_set_path = './dataset/trainset'
    with open(train_set_path, 'r') as f_index:
        lines = f_index.readlines()
        for line in tqdm(lines, desc="Training Data", unit="file"):
            ## 问题1: 训练集大小对性能的影响
            if random.random() > sample_rate:
                continue

            label, file_path = line.strip().split(' ')
            file_path = 'trec06p' + file_path[2:]

            ## Read each email file in train set
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f_email:
                text = f_email.read()
                ## Extract all words
                words = text.strip().split(' ')

                ## 问题3: 特征设计
                received_from = re.findall(r'Received: from\s+(\S+)', text)
                if received_from:
                    next_word = received_from[0].split()[0]
                    words.append(next_word)

                ## Add word frequency to word_freq dictionary
                word_freq[label].update(words)
                total_freq[label] += len(words)


    with open('frequency/word_freq.json', 'w') as wf:
        json.dump(word_freq, wf, ensure_ascii=False, indent=2)
    with open('frequency/total_freq.json', 'w') as tf:
        json.dump(total_freq, tf, ensure_ascii=False, indent=2)


# 3: Test data and Evaluate model using five-fold cross-validation
def evaluate_model(laplace=False, l_index=1):
    ## Function to create folds for cross-validation
    def create_folds(dataset, k=5):
        folds = []
        fold_size = len(dataset) // k
        for i in range(k):
            start = i * fold_size
            end = (i + 1) * fold_size if i < k - 1 else len(dataset)
            fold = dataset[start:end]
            folds.append(fold)
        return folds

    # Load data from the test set
    test_set_path = './dataset/testset'
    with open(test_set_path, 'r') as f_index:
        lines = f_index.readlines()
        dataset = [line.strip().split(' ') for line in lines]

    with open('frequency/word_freq.json') as wf:
        word_freq = json.load(wf)
    with open('frequency/total_freq.json') as tf:
        total_freq = json.load(tf)

    folds = create_folds(dataset)
    accuracies, precisions, recalls, f1_scores = [], [], [], []

    for fold in tqdm(folds, desc="Cross-Validation Folds"):
        true_num = 0
        total_num = 0
        true_positive = 0
        false_positive = 0
        false_negative = 0

        # Evaluate the model on each fold
        for label, file_path in fold:
            file_path = 'trec06p' + file_path[2:]
            max_prob = -math.inf
            tmp_label = ''

            for y in total_freq.keys():
                prob = math.log(total_freq[y] / float(sum(total_freq.values())))
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f_email:
                    text = f_email.read()
                    words = text.strip().split(' ')

                    ## 问题3: 特征设计
                    received_from = re.findall(r'Received: from\s+(\S+)', text)
                    if received_from:
                        next_word = received_from[0].split()[0]
                        words.append(next_word)

                    ## Laplace Smoothing
                    if laplace:
                        prob += sum(math.log((word_freq[y].get(word, 0) + l_index) / (total_freq[y] + len(word_freq[y]) * l_index)) for word in words)
                    else:
                        prob += sum(math.log(word_freq[y].get(word, 1) / total_freq[y]) for word in words)
                if prob > max_prob:
                    max_prob = prob
                    tmp_label = y
            total_num += 1
            true_num += tmp_label == label
            if tmp_label == label:
                true_positive += 1
            else:
                false_negative += 1

        accuracy = true_num / total_num
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) != 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

    avg_accuracy = sum(accuracies) / len(accuracies)
    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)
    avg_f1_score = sum(f1_scores) / len(f1_scores)

    return avg_accuracy, avg_precision, avg_recall, avg_f1_score



if __name__ == '__main__':
    ## Parameters
    train_prob = 0.9
    random_seed = 1
    sample_rate = 1
    laplace = True
    ## Model Functions
    dataset(train_prob, random_seed)
    train(sample_rate)
    avg_accuracy, avg_precision, avg_recall, avg_f1_score = evaluate_model(laplace)
    
    print("Average Accuracy:", avg_accuracy)
    print("Average Precision:", avg_precision)
    print("Average Recall:", avg_recall)
    print("Average F1 Score:", avg_f1_score)
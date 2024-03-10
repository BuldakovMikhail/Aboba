import joblib as jb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import ndcg_score
from sklearn.preprocessing import StandardScaler

import itertools
from sklearn import svm, linear_model


def show_menu():
    print(
        """
          Меню:
          1 --- Посчитать метрику
          2 --- Получить предсказания
          0 --- Выйти
          
          """
    )


def transform_pairwise(X, y):
    X_new = []
    y_new = []
    y = np.asarray(y)
    if y.ndim == 1:
        y = np.c_[y, np.ones(y.shape[0])]
    comb = itertools.permutations(range(X.shape[0]), 2)
    for k, (i, j) in enumerate(comb):
        if y[i, 0] == y[j, 0] or y[i, 1] != y[j, 1]:
            continue
        X_new.append(X[i] - X[j])
        y_new.append(np.sign(y[i, 0] - y[j, 0]))
        if y_new[-1] != (-1) ** k:
            y_new[-1] = -y_new[-1]
            X_new[-1] = -X_new[-1]
    return np.asarray(X_new), np.asarray(y_new).ravel()


class RankSVM(svm.LinearSVC):
    def fit(self, ids, X, y):
        ids = np.array(ids)
        unique = np.unique(ids)
        res_x = []
        res_y = []
        for i in unique:
            X_trans, y_trans = transform_pairwise(X[ids == i], y[ids == i])
            if not np.any(X_trans):
                continue
            res_x.append(X_trans)
            res_y.append(y_trans)

        x_trans_ac = np.concatenate(res_x)
        y_trans_ac = np.concatenate(res_y)

        super(RankSVM, self).fit(x_trans_ac, y_trans_ac)
        return self

    def decision_function(self, X):
        return np.dot(X, self.coef_.ravel())

    def predict(self, ids, X):
        if hasattr(self, "coef_"):
            ids = np.array(ids)
            unique = np.unique(ids)
            res = []
            for i in unique:
                res += list(np.dot(X[ids == i], self.coef_.ravel()))
            return np.array(res)
        else:
            raise ValueError("Must call fit() prior to predict()")

    def score(self, X, y):
        X_trans, y_trans = transform_pairwise(X, y)
        return np.mean(super(RankSVM, self).predict(X_trans) == y_trans)


model = jb.load("model_rank_svm_without_ros.pk1")


def count_metric(preds, y_true):
    return ndcg_score([y_true], [preds])


def my_dcg(y_true, predicted):
    sorted = np.argsort(predicted)[::-1]
    res = 0
    for i in range(len(sorted)):
        res += y_true[sorted[i]] / np.log2(i + 2)
    return res


def count_metric_gr(idx, preds, y_true):
    pred_gr = {}
    true_gr = {}

    for i in range(len(idx)):
        if idx[i] not in pred_gr.keys():
            pred_gr[idx[i]] = [preds[i]]
            true_gr[idx[i]] = [y_true[i]]
        else:
            pred_gr[idx[i]].append(preds[i])
            true_gr[idx[i]].append(y_true[i])

    res = 0
    count = 0

    for k, v in pred_gr.items():
        count_rel = sum(true_gr[k])
        if count_rel == 0:
            continue
        dcg = my_dcg(np.array(true_gr[k]), np.array(v))
        idcg = 0
        for i in range(count_rel):
            idcg += 1 / np.log2(i + 2)

        res += dcg / idcg
        count += 1

    return res / count


def run_metrics(ids, y_preds, y_true):
    m_all = count_metric(y_preds, y_true)
    m_gr = count_metric_gr(ids, y_preds, y_true)

    return (m_all, m_gr)


def count_metric_f(file):
    df = pd.read_csv(file)

    X = df.drop("target", axis=1)
    y = df["target"]
    scaler = StandardScaler()

    x_test = X.drop("search_id", axis=1)
    ids_test = list(X["search_id"])

    x_test_scaled = scaler.fit_transform(x_test)

    y_preds = model.predict(ids_test, x_test_scaled)
    return run_metrics(ids_test, y_preds, y)


def get_prediction(file):
    df = pd.read_csv(file)
    scaler = StandardScaler()

    X = scaler.fit_transform(df)

    y_preds = model.predict(np.ones(shape=len(X)), X)

    return np.argsort(y_preds), df.iloc[np.argsort(y_preds)]


def main():
    print("---------------------------------------------------------------------")
    print("                         Model:Ranking                               ")
    print("---------------------------------------------------------------------")

    while True:
        show_menu()
        opt = int(input("Введит пункт меню: "))
        if opt == 0:
            return
        elif opt == 1:
            metric_count = input("Файл для подсчета метрик: ")
            m_all, m_gr = count_metric_f(metric_count)

            print("Metric all: ", m_all)
            print("Metric group: ", m_gr)
        elif opt == 2:
            file_for_prediction = input(
                "Файл для предсказаний (без таргета и для конкретной группы): "
            )
            args, vals = get_prediction(file_for_prediction)
            print("Ranked df: ")
            print(vals)
            print("Perms: ")
            print(args)
        else:
            print("Неверный пункт")


if __name__ == "__main__":
    main()

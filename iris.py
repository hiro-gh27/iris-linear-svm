import pandas
import numpy

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVC
from mlxtend.plotting import plot_decision_regions
from sklearn import datasets
import logging


LOG_LEVEL = logging.DEBUG
FORMATTER = '%(asctime)s: %(levelname)s (%(filename)s:%(lineno)d)\n %(message)s'
logging.basicConfig(level=LOG_LEVEL, format=FORMATTER)
logger = logging.getLogger("iris.py")

principal_a = 'principal-a'
principal_b = 'principal-b'


def main():
    scr = "./IrisData.csv"
    df = pandas.read_csv(scr, encoding="shift-jis"
                         , names=['target', 'sepal length', 'sepal width', 'petal length', 'petal width'])
    logger.debug(df)

    features = ['sepal length', 'sepal width', 'petal length', 'petal width']
    x = df.loc[:, features].values
    y = df.loc[:, ['target']].values
    sub = {'セトーサ': 'setosa', 'ベルシカラー': 'versicolor-virginica', 'バージニカ': 'versicolor/virginica'}
    sub = {'セトーサ': 0, 'ベルシカラー': 1, 'バージニカ': 1}
    y = [[sub.get(x, x) for x in inner] for inner in y]
    logger.debug(y)
    y = pandas.DataFrame(data=y, columns=['target'])
    logger.debug(x)


    x = StandardScaler().fit_transform(x)
    logger.debug(x)

    pca = PCA(n_components=2)
    principal_component = pca.fit_transform(x)
    logger.debug(principal_component)

    principal_data_frame = pandas.DataFrame(data=principal_component, columns=[principal_a, principal_b])
    logger.debug(principal_data_frame)

    x_train, x_test, y_train, y_test = train_test_split(principal_data_frame
                                                        , y['target'], test_size=0.8)

    training_model = SVC(kernel='linear', random_state=None)
    training_model.fit(x_train, y_train)

    predict_result = training_model.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, predict_result)
    print("accuracy: ", accuracy)

    plt.style.use('ggplot')
    fig = plt.figure(figsize=(10, 10))
    x_combined = numpy.vstack((x_train, x_test))
    y_combined = numpy.hstack((y_train, y_test))
    # print(x_combined)
    # print(y_combined)
    plot_decision_regions(x_combined, y_combined, clf=training_model, )
    # principal_data_frame.plot(kind='scatter', x=principal_a, y=principal_b)
    plt.show()
    fig.savefig('figure.png')


def n_pca(n, scr):
    pca = PCA(n_components=n)
    principal_components = pca.fit_transform(scr)
    result = None

    if n == 1:
        zero_array = numpy.zeros(150)
        zero_data_frame = pandas.DataFrame(data=zero_array, columns=[principal_b])
        principals_data_frame = pandas.DataFrame(data=principal_components, columns=[principal_a])
        result = pandas.concat([principals_data_frame, zero_data_frame], axis=1)

    print(result)
    return result


def test_iris():
    iris_data = datasets.load_iris()
    y = iris_data.target
    print(y)


if __name__ == '__main__':
    main()





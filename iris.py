import pandas
import numpy

import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVC
from mlxtend.plotting import plot_decision_regions
import logging


LOG_LEVEL = logging.WARNING
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
    y = pandas.DataFrame(data=y, columns=['target'])
    x = StandardScaler().fit_transform(x)

    principal_data_frame = n_pca(2, x)
    x_train, x_test, y_train, y_test = train_test_split(principal_data_frame
                                                        , y['target'], test_size=0.8)

    training_model = SVC(kernel='linear', random_state=None)
    training_model.fit(x_train, y_train)

    predict_result = training_model.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, predict_result)
    print("accuracy: ", accuracy)

    x_combined = numpy.vstack((x_train, x_test))
    y_combined = numpy.hstack((y_train, y_test))
    convert_figure(x_combined, y_combined, training_model, 10)


def n_pca(n, scr):
    pca = PCA(n_components=n)
    principal_components = pca.fit_transform(scr)
    result = None
    if n == 1:
        zero_array = numpy.zeros(150)
        zero_data_frame = pandas.DataFrame(data=zero_array, columns=[principal_b])
        principals_data_frame = pandas.DataFrame(data=principal_components, columns=[principal_a])
        result = pandas.concat([principals_data_frame, zero_data_frame], axis=1)
    if n == 2:
        result = pandas.DataFrame(data=principal_components, columns=[principal_a, principal_b])
    logger.debug(result)
    return result


# LDAがわからないため，PCAを利用した．
def n_lda(n, src_x, scr_y, ):
    lda = LinearDiscriminantAnalysis(n_components=n)
    principal_component = lda.fit_transform(src_x, scr_y)
    principal_data_frame = pandas.DataFrame(data=principal_component, columns=[principal_a])
    print(principal_component)
    zero_array = numpy.zeros(len(src_x))
    zero_data_frame = pandas.DataFrame(data=zero_array, columns=['zero-padding'])
    lda_result_data_frame = pandas.concat([principal_data_frame, zero_data_frame], axis=1)
    print(lda_result_data_frame)
    return lda_result_data_frame


def convert_figure(x, y, classifier, size):
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(size, size))
    plot_decision_regions(x, y, clf=classifier)
    now_datetime = datetime.now()
    result_png_file_name = now_datetime.strftime('%Y-%m-%d%H:%M:%S')+'.png'
    plt.show()
    fig.savefig(result_png_file_name)
    logger.debug(result_png_file_name)
    return


if __name__ == '__main__':
    main()





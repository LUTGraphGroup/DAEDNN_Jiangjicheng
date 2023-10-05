import time
import numpy as np
from keras import Sequential
from keras.models import Model
from keras.layers import Dense, Input, GaussianNoise, Activation, Dropout, Multiply
from keras.optimizers import SGD
from keras.utils import np_utils, to_categorical
from matplotlib import pyplot as plt, pyplot
from numpy import matlib as mb, interp
from sklearn import preprocessing
from sklearn.decomposition import PCA, NMF
from sklearn.metrics import auc, precision_recall_curve, roc_curve, accuracy_score, precision_score, recall_score, \
    f1_score
import warnings
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

#去噪编码器
def DAE(x_train):
    encoding_dim = 128
    # input_img = Input(shape=(1162,))
    input_img = Input(shape=(626,))

    # Add Gaussian noise to input
    noise = GaussianNoise(0.1)(input_img)

    # encoder layers
    encoded = Dense(512, activation='relu')(noise)
    encoded = Dense(256, activation='relu')(encoded)
    encoder_output = Dense(encoding_dim)(encoded)

    # Attention mechanism
    attention_probs = Dense(128, activation='sigmoid')(encoder_output)
    attention_mul = Multiply()([encoder_output, attention_probs])

    # decoder layers
    decoded = Dense(256, activation='relu')(attention_mul)
    decoded = Dense(512, activation='relu')(decoded)
    decoded = Dense(626, activation='tanh')(decoded)
    # decoded = Dense(1162, activation='tanh')(decoded)

    autoencoder = Model(input=input_img, output=decoded)
    encoder = Model(input=input_img, output=encoder_output)
    autoencoder.compile(optimizer='adam', loss='mse')

    # training
    autoencoder.fit(x_train, x_train, epochs=20, batch_size=128, shuffle=True)
    encoded_imgs = encoder.predict(x_train)
    return encoder_output, encoded_imgs


# def PCA_model(x_train):
#     n_components = 128
#     # 创建PCA模型并拟合训练数据
#     pca = PCA(n_components=n_components)
#     pca.fit(x_train)
#
#     # 使用PCA模型对训练数据进行转换
#     encoded_imgs = pca.transform(x_train)
#
#     return pca, encoded_imgs

# from sklearn.decomposition import FastICA

# def ICA_model(x_train):
#     n_components = 128
#     # Create ICA model and fit training data
#     ica = FastICA(n_components=n_components)
#     encoded_imgs = ica.fit_transform(x_train)
#
#     return ica, encoded_imgs
# def NMF_model(x_train):
#     # Create NMF model and fit training data
#     n_components = 128
#     nmf = NMF(n_components=n_components)
#     encoded_imgs = nmf.fit_transform(x_train)
#
#     return nmf, encoded_imgs

def DNN():
    model = Sequential()
    model.add(Dense(input_dim=128, output_dim=500))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    model.add(Dense(input_dim=500, output_dim=500, init='glorot_normal'))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    model.add(Dense(input_dim=500, output_dim=300))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    model.add(Dense(input_dim=300, output_dim=2))
    model.add(Activation('sigmoid'))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    return model

def load_data():
    # MM = np.loadtxt(r".\data\MM.txt")
    # DD = np.loadtxt(r".\data\DD.txt")
    # A = np.loadtxt(r".\data\miRNA-disease association.txt", dtype=int, delimiter="\t")
    MM = np.loadtxt('integrated miRNA similarity.txt')
    DD = np.loadtxt('integrated disease similarity.txt')
    A = np.loadtxt('association matrix.txt')
    # MS = np.where(MM > 0.5, 1, 0)
    # DS = np.where(DD > 0.5, 1, 0)
    # mm = np.repeat(MS, repeats=374, axis=0)
    # dd = mb.repmat(DS, 788, 1)
    # mm = np.repeat(MM, repeats=374, axis=0)
    # dd = mb.repmat(DD, 788, 1)
    # H = np.concatenate((dd, mm), axis=1)  # (294712,1162)
    #
    # label = A.reshape((294712, 1))
    mm = np.repeat(MM, repeats=149, axis=0)
    dd = mb.repmat(DD, 477, 1)
    H = np.concatenate((dd, mm), axis=1)  # (294712,1162)

    label = A.reshape((71073, 1))
    # matadj_mirna = np.array(MM)
    # matadj_disease = np.array(DD)
    # mirna_disease_matrix = np.array(A)
    # #计算mirna和疾病节点的邻接矩阵
    # H_adjmat_1 = np.hstack((matadj_mirna, mirna_disease_matrix))
    # H_adjmat_2 = np.hstack((mirna_disease_matrix.transpose(), matadj_disease))
    # H = np.vstack((H_adjmat_1, H_adjmat_2))
    # label = np.random.randint(0, 2, size=(1162, 1)).astype(np.float32)
    return H, label

# one-hot
def preprocess_labels(labels, encoder=None, categorical=True):
    if not encoder:
        encoder = preprocessing.LabelEncoder()
        encoder.fit(labels)
    y = encoder.transform(labels).astype(np.int32)
    if categorical:
        y = np_utils.to_categorical(y)
    return y, encoder

def transfer_label_from_prob(proba):
    label = [1 if val >= 0.5 else 0 for val in proba]
    return label


def DAEDNN():
    H, label = load_data()
    y, encoder = preprocess_labels(label)
    num = np.arange(len(y))
    y = y[num]

    encoder, H_data = DAE(H)
    # encoder, H_data = PCA_model(H)
    # encoder, H_data = NMF_model(H)

    t = 0
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    KFold = 5
    aucs = []
    performance = []
    AUC_list = []
    AUPRC_list = []
    Accuracy_list = []
    Precision_list = []
    Recall_list = []
    F1_list = []

    for fold in range(KFold):
        train = np.array([x for i, x in enumerate(H_data) if i % KFold != fold])
        test = np.array([x for i, x in enumerate(H_data) if i % KFold == fold])
        train_label = np.array([x for i, x in enumerate(y) if i % KFold != fold])
        test_label = np.array([x for i, x in enumerate(y) if i % KFold == fold])

        real_labels = []
        for val in test_label:
            if val[0] == 1:
                real_labels.append(0)
            else:
                real_labels.append(1)

        train_label_new = []
        for val in train_label:
            if val[0] == 1:
                train_label_new.append(0)
            else:
                train_label_new.append(1)

        prefilter_train = train
        prefilter_test = test

        # clf = XGBClassifier(n_estimators=100, learning_rate=0.3)
        # from sklearn.tree import DecisionTreeClassifier
        #
        # clf = DecisionTreeClassifier(max_depth=5, random_state=42)
        # clf.fit(prefilter_train, train_label_new)  # Training
        # ae_y_pred_prob = clf.predict_proba(prefilter_test)[:, 1]  # testing
        # proba = transfer_label_from_prob(ae_y_pred_prob)
        # from sklearn.linear_model import LogisticRegression
        #
        # clf = LogisticRegression(random_state=42)
        # clf.fit(prefilter_train, train_label_new)  # Training
        # ae_y_pred_prob = clf.predict_proba(prefilter_test)[:, 1]  # testing
        # proba = transfer_label_from_prob(ae_y_pred_prob)
        # from sklearn.neighbors import KNeighborsClassifier
        #
        # clf = KNeighborsClassifier(n_neighbors=5)
        # clf.fit(prefilter_train, train_label_new)  # Training
        # ae_y_pred_prob = clf.predict_proba(prefilter_test)[:, 1]  # testing
        # proba = transfer_label_from_prob(ae_y_pred_prob)

        # clf = XGBClassifier(n_estimators=100, learning_rate=0.01)
        # clf = svm.SVC(kernel='linear', probability=True)
        # clf.fit(prefilter_train, train_label_new)  # Training
        # ae_y_pred_prob = clf.predict_proba(prefilter_test)[:, 1]  # testing
        # proba = transfer_label_from_prob(ae_y_pred_prob)
        # from sklearn.naive_bayes import GaussianNB
        #
        # clf = GaussianNB()
        # clf.fit(prefilter_train, train_label_new)  # Training
        # ae_y_pred_prob = clf.predict_proba(prefilter_test)[:, 1]  # testing
        # proba = transfer_label_from_prob(ae_y_pred_prob)
        # from sklearn.ensemble import RandomForestClassifier
        #
        # clf = RandomForestClassifier(n_estimators=100)
        # clf.fit(prefilter_train, train_label_new)  # Training
        # ae_y_pred_prob = clf.predict_proba(prefilter_test)[:, 1]  # testing
        # proba = transfer_label_from_prob(ae_y_pred_prob)

        model_DNN = DNN()
        train_label_new = to_categorical(train_label_new, num_classes=2)
        model_DNN.fit(prefilter_train, train_label_new, batch_size=200, nb_epoch=20, shuffle=True)
        ae_y_pred_prob = model_DNN.predict_proba(prefilter_test, batch_size=200, verbose=True)[:, 1]

        # ae_y_pred_prob = model_DNN.predict_proba(H_data, batch_size=200, verbose=True)[:, 1]
        # print(ae_y_pred_prob)
        # np.savetxt("predict11.txt", ae_y_pred_prob, delimiter="\t", fmt='%.4f')

        # clf.fit(prefilter_train, train_label_new)  # Training
        # ae_y_pred_prob = clf.predict_proba(prefilter_test)[:, 1]  # testing
        # proba = transfer_label_from_prob(ae_y_pred_prob)

        # acc, precision, MCC, f1_score = calculate_performace(len(real_labels), proba, real_labels)
        # precision, recall, _ = precision_recall_curve(real_labels, ae_y_pred_prob)
        # average_precision = np.average(precision)
        Precision, Recall, _ = precision_recall_curve(real_labels, ae_y_pred_prob)
        AUPRC = auc(Recall, Precision)
        AUPRC_list.append(AUPRC)
        A = accuracy_score(real_labels, ae_y_pred_prob.round())
        Accuracy_list.append(A)
        P = precision_score(real_labels, ae_y_pred_prob.round())
        Precision_list.append(P)
        R = recall_score(real_labels, ae_y_pred_prob.round())
        Recall_list.append(R)
        F1 = f1_score(real_labels, ae_y_pred_prob.round())
        F1_list.append(F1)
        fpr, tpr, auc_thresholds = roc_curve(real_labels, ae_y_pred_prob)

        auc_score = auc(fpr, tpr)
        aucs.append(auc_score)

        performance.append([auc_score])

        t = t + 1
        np.savetxt("DAEDNN_fpr_" + str(t) + ".txt", fpr)
        np.savetxt("DAEDNN_tpr_" + str(t) + ".txt", tpr)

        pyplot.plot(fpr, tpr, linewidth=1, alpha=0.5, label='ROC fold %d (AUC = %0.4f)' % (t, auc_score))

        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0


    mean_tpr /= KFold
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    np.savetxt('DNN_mean_fpr.csv', mean_fpr, delimiter=',')
    np.savetxt('DNN_mean_tpr.csv', mean_tpr, delimiter=',')
    std_auc = np.std(aucs)
    print('std_auc=', std_auc)
    print('mean_AUC=', mean_auc)
    pyplot.plot(mean_fpr, mean_tpr, 'black', linewidth=1.5, alpha=0.8,
                label='Mean ROC(AUC = %0.4f)' % mean_auc)

    pyplot.legend()

    plt.savefig('5-fold CV DAEDNN(AUC = %0.4f).png' % mean_auc, dpi=300)

    pyplot.show()
    # 绘制PR曲线
    plt.figure()
    plt.step(Recall, Precision, color='b', alpha=0.2, where='post')
    plt.fill_between(Recall, Precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve')
    pyplot.plot(Recall, Precision, linewidth=1, alpha=0.5, label='ROC fold %d (AUPRC = %0.4f)' % (t, AUPRC))
    plt.savefig('PR_curve.png', dpi=300)
    plt.show()
    print('--------------------------------------------------------')
    print("AUPRC average is", sum(AUPRC_list) / len(AUPRC_list))
    print("the average of Accuracy is ", sum(Accuracy_list) / len(Accuracy_list))
    print("the average of Precision is ", sum(Precision_list) / len(Precision_list))
    print("the average of Recall is ", sum(Recall_list) / len(Recall_list))
    print("the average of F1 is ", sum(F1_list) / len(F1_list))
if __name__ == "__main__":
    time_start = time.time()

    DAEDNN()
    time_end = time.time()
    print('time cost', time_end - time_start, 's')

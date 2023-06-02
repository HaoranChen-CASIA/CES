# Combine CA and SA as one category

from Judge.test_4_category import *

from sklearn.neighbors import KNeighborsClassifier as KNNC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


def generate_classify_dataset(T4C, root_dir, seq_length, mode='CES', z_fill=4):
    print('Current Mode: ', mode)
    points = []
    for cata in ['/CA/', '/SA/', '/PA/', '/WA/']:
        ref_dir = root_dir + cata + 'ref/'
        tgt_dir = root_dir + cata + 'mov/'
        pt = []
        for idx in range(0, seq_length, 1):
            img_r = cv2.imread(ref_dir + str(idx).zfill(z_fill) + '.png', 0)
            img_m = cv2.imread(tgt_dir + str(idx).zfill(z_fill) + '.png', 0)
            sim = T4C.compute_similarity(img_r, img_m, mode=mode)
            pt.append(sim)
        T4C.flier_counter(pt)
        points.append(pt)

    # save points
    with open(root_dir + '/points_' + mode + '.pkl', 'wb') as f:
        pickle.dump(points, f)


def combine_all_dataset(root_dir, seq_length, save=True):
    # step 1: load point dataset
    for mode in ['CES']:
        with open(root_dir + '/points_' + mode + '.pkl', 'rb') as f:
            points = pickle.load(f)
        flatten_points = np.array(list(chain(*points)))
        if mode == 'CES':
            data = np.resize(flatten_points, (4*seq_length, 1))
        else:
            data = np.concatenate((data, np.resize(flatten_points, (4*seq_length, 1))), axis=1)

    label = np.ones(len(flatten_points), dtype='int').ravel()
    label[:2*seq_length] = 0  # combine label 'CA' and 'SA' as one category
    label[2*seq_length:3*seq_length] = 1  # label 'PA'
    label[3*seq_length:] = 2  # label 'WA'

    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.3)

    if save:
        data_path = root_dir + '/data_label_3cate/'
        os.makedirs(data_path, exist_ok=True)
        with open(data_path + 'x_train.pkl', 'wb') as f4:
            pickle.dump(x_train, f4)
        with open(data_path + 'x_test.pkl', 'wb') as f5:
            pickle.dump(x_test, f5)
        with open(data_path + 'y_train.pkl', 'wb') as f6:
            pickle.dump(y_train, f6)
        with open(data_path + 'y_test.pkl', 'wb') as f7:
            pickle.dump(y_test, f7)

    return x_train, x_test, y_train, y_test


def load_dataset(root_dir):
    data_path = root_dir + '/data_label_3cate/'

    with open(data_path + 'x_train.pkl', 'rb')as f4:
        x_train = pickle.load(f4)
    with open(data_path + 'x_test.pkl', 'rb')as f5:
        x_test = pickle.load(f5)
    with open(data_path + 'y_train.pkl', 'rb')as f6:
        y_train = pickle.load(f6)
    with open(data_path + 'y_test.pkl', 'rb')as f7:
        y_test = pickle.load(f7)
    return x_train, x_test, y_train, y_test


def train_and_show_acc(x_train, y_train, x_test, y_test, score, k, show=False):
    knn = KNNC(n_neighbors=k)
    knn.fit(x_train, y_train)
    y_pre = knn.predict(x_test)
    score.append(knn.score(x_test, y_test))
    if show:
        print('\nk = {}, acc = {}'.format(k, knn.score(x_test, y_test)))
    return knn, y_pre


def get_knn_acc(y_test, y_pre, label):
    cnt = 0  # Correct
    assert y_test.shape[0] == y_pre.shape[0]
    for i in range(len(y_test)):
        if y_test[i] == y_pre[i] and y_test[i] == label:
            cnt += 1
    return cnt/np.sum(y_test == label)


def draw_confusion_metrics(root_dir, C, mode):
    indices = range(len(C))
    classes = ['CA*', 'PA', 'WA']
    pred_classes = ['CA*\'', 'PA\'', 'WA\'']
    # plt.matshow(C, cmap=plt.cm.Blues)  # change color here
    plt.matshow(C, cmap=plt.cm.Oranges)  # change color here
    # plt.colorbar()

    for i in range(len(C)):
        for j in range(len(C)):
            plt.annotate(C[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center', fontsize=20)

    # plt.tick_params(labelsize=15) # set font size
    plt.xticks(indices, pred_classes)
    plt.yticks(indices, classes)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    os.makedirs(root_dir + '/Confusion_Metrics_3cate', exist_ok=True)
    plt.savefig(root_dir + '/Confusion_Metrics_3cate/' + mode + '.tiff')


def get_F1(C):
    # input the confusion metric and return precision, recall and F1 score
    idx = 0
    for cate in ['CA+SA', 'PA', 'WA']:
        precision = C[idx, idx] / np.sum(C[:, idx])
        recall = C[idx, idx] / np.sum(C[idx, :])
        F1 = 2 * (precision * recall) / (precision + recall)
        print(cate + ' F1 score {}, precision {}, recall {}'.format(F1, precision, recall))
        idx += 1

        
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # avoid traffic jam
    T4C = Test4Category()
    root_dir = '../data/EXP2_FlyEM_BS/Dataset_z32nm'
    seq_length = 640

    x_train_src, x_test_src, y_train, y_test = combine_all_dataset(root_dir, seq_length, save=True)
    # x_train_src, x_test_src, y_train, y_test = load_dataset(root_dir)

    idx = 0
    for mode in ['CES']:
        print('\nmode={}'.format(mode))
        x_train = x_train_src[:, idx].reshape(-1, 1)
        x_test = x_test_src[:, idx].reshape(-1, 1)
        score = []
        k_s, k_e = 1, 16
        for k in range(k_s, k_e):
            knn, _ = train_and_show_acc(x_train, y_train, x_test, y_test, score, k)

        score = np.array(score)
        print('max acc at k={}'.format(k_s + np.where(score == score.max())[0][0]))
        k = k_s + np.where(score == score.max())[0][0]

        # record best performance
        score = []
        knn, y_pre = train_and_show_acc(x_train, y_train, x_test, y_test, score, k, show=True)
        C = confusion_matrix(y_test, y_pre, labels=[0, 1, 2])
        get_F1(C)
        draw_confusion_metrics(root_dir, C, mode)
        idx += 1

    print('done')



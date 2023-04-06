
def example():
    origdata = pd.read_csv("/content/drive/MyDrive/Data_Active_Learning/Iris.csv")
    origdata[:10]

    k1, k2 = 'PetalLengthCm', 'PetalWidthCm'
    data = origdata[[k1, k2, 'Species']].copy()
    data[:10]

    X = data[[k1, k2]]
    y = data['Species']
    print('Classes:')
    print(y.unique(), '\n\n\n')

    y[y=='Iris-setosa'] = 0
    y[y=='Iris-versicolor'] = 1
    y[y=='Iris-virginica'] = 2

    plt.figure()
    setosa = y == 0
    versicolor = y == 1
    virginica = y == 2

    plt.scatter(X[k1][versicolor], X[k2][versicolor], c='r')
    plt.scatter(X[k1][virginica], X[k2][virginica], c='c')
    plt.xlabel(k1)
    plt.ylabel(k2)
    plt.show()

    X1 = X[y != 0]
    y1 = y[y != 0]
    X1[:5]

    X1 = X1.reset_index(drop=True)
    y1 = y1.reset_index(drop=True)
    y1 -= 1
    print(y1.unique())
    X1[:5]

    fig = plt.figure()

    plt.scatter(X1[k1][y1 == 0], X1[k2][y1 == 0], c='r')
    plt.scatter(X1[k1][y1 == 1], X1[k2][y1 == 1], c='c')

    plt.xlabel(k1)
    plt.ylabel(k2)
    fig.savefig('main.jpg', dpi=100)
    plt.show()

    y1 = y1.astype(dtype=np.uint8)
    clf0 = LinearSVC()
    clf0.fit(X1, y1)
    LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
              intercept_scaling=1, loss='squared_hinge', max_iter=1000,
              multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
              verbose=0)
    print(clf0.coef_)
    print(clf0.intercept_)

    xmin, xmax = X1[k1].min(), X1[k1].max()
    ymin, ymax = X1[k2].min(), X1[k2].max()
    stepx = (xmax - xmin) / 99
    stepy = (ymax - ymin) / 99
    a0, b0, c0 = clf0.coef_[0, 0], clf0.coef_[0, 1], clf0.intercept_
    # Formula for reference
    # a*x + b*y + c = 0
    # y = -(a*x + c)/b

    lx0 = [xmin + stepx * i for i in range(100)]
    ly0 = [-(a0 * lx0[i] + c0) / b0 for i in range(100)]

    plt.figure()

    plt.scatter(X1[k1][y1 == 0], X1[k2][y1 == 0], c='r')
    plt.scatter(X1[k1][y1 == 1], X1[k2][y1 == 1], c='c')

    plt.plot(lx0, ly0, c='m')

    plt.xlabel(k1)
    plt.ylabel(k2)

    plt.show()

    X_pool, X_test, y_pool, y_test = train_test_split(X1, y1, test_size=0.2, random_state=1)
    X_pool, X_test, y_pool, y_test = X_pool.reset_index(drop=True), X_test.reset_index(drop=True), y_pool.reset_index(
        drop=True), y_test.reset_index(drop=True)
    # random state 1 5 iterations
    # random state 2 20 iterations

    clf0.decision_function(X_pool.iloc[6:8])

    def find_most_ambiguous(clf, unknown_indexes):
        ind = np.argmin(np.abs(
            list(clf0.decision_function(X_pool.iloc[unknown_indexes]))
        ))
        return unknown_indexes[ind]

    def plot_svm(clf, train_indexes, unknown_indexes, new_index=False, title=False, name=False):
        X_train = X_pool.iloc[train_indexes]
        y_train = y_pool.iloc[train_indexes]
        X_new = 0

        X_unk = X_pool.iloc[unknown_indexes]

        if new_index:
            X_new = X_pool.iloc[new_index]

        a, b, c = clf.coef_[0, 0], clf.coef_[0, 1], clf.intercept_
        # Straight Line Formula
        # a*x + b*y + c = 0
        # y = -(a*x + c)/b

        lx = [xmin + stepx * i for i in range(100)]
        ly = [-(a * lx[i] + c) / b for i in range(100)]

        fig = plt.figure(figsize=(9, 6))

        # plt.scatter(x[k1][setosa], x[k2][setosa], c='r')
        plt.scatter(X_unk[k1], X_unk[k2], c='k', marker='.')
        plt.scatter(X_train[k1][y_train == 0], X_train[k2][y_train == 0], c='r', marker='o')
        plt.scatter(X_train[k1][y_train == 1], X_train[k2][y_train == 1], c='c', marker='o')

        plt.plot(lx, ly, c='m')
        plt.plot(lx0, ly0, '--', c='g')

        if new_index:
            plt.scatter(X_new[k1], X_new[k2], c='y', marker="*", s=125)
            # plt.scatter(X_new[k1], X_new[k2], c='y', marker="*", s=125)
            # plt.scatter(X_new[k1], X_new[k2], c='y', marker="*", s=125)
            # plt.scatter(X_new[k1], X_new[k2], c='m', marker="*", s=125)
            # plt.scatter(X_new[k1], X_new[k2], c='m', marker="*", s=125)

        if title:
            plt.title(title)

        plt.xlabel(k1)
        plt.ylabel(k2)

        if name:
            fig.set_size_inches((9, 6))
            plt.savefig(name, dpi=100)

        plt.show()

        train_indexes = list(range(10))
        unknown_indexes = list(range(10, 80))
        X_train = X_pool.iloc[train_indexes]
        y_train = y_pool.iloc[train_indexes]
        clf = LinearSVC()
        clf.fit(X_train, y_train)

        # folder = "rs1it5/"
        folder = "rs2it20/"
        # folder = "rs1it20/"

        try:
            os.mkdir(folder)
        except:
            pass

        filenames = ["ActiveLearningTitleSlide2.jpg"] * 2

        title = "Beginning"
        # name = folder + ("rs1it5_0a.jpg")
        name = folder + ("rs2it20_0a.jpg")
        plot_svm(clf, train_indexes, unknown_indexes, False, title, name)

        # filenames.append(name)

        n = find_most_ambiguous(clf, unknown_indexes)
        unknown_indexes.remove(n)
        # print(n)

        title = "Iteration 0"
        name = folder + ("rs1it5_0b.jpg")
        # name = folder + ("rs2it20_0b.jpg")
        filenames.append(name)
        plot_svm(clf, train_indexes, unknown_indexes, n, title, name)

        train_indexes = list(range(10))
        unknown_indexes = list(range(10, 80))
        X_train = X_pool.iloc[train_indexes]
        y_train = y_pool.iloc[train_indexes]
        clf = LinearSVC()
        print(y_train.to_list())
        clf.fit(X_train, y_train.to_list())

        # folder = "rs1it5/"
        folder = "test/"
        # folder = "rs1it20/"

        try:
            os.mkdir(folder)
        except:
            pass

        filenames = ["ActiveLearningTitleSlide2.jpg"] * 2

        title = "Beginning"
        # name = folder + ("rs1it5_0a.jpg")
        name = folder + ("test_0a.jpg")
        # plot_svm(clf, train_indexes, unknown_indexes, False, title, name)

        # filenames.append(name)

        n = find_most_ambiguous(clf, unknown_indexes)

        print(X_pool.iloc[n])
        label_n = input(
            "What label would you give this datapoint (also shown by the star)? \n 0 for not interesting, 1 for interesting. \n")
        print("you have chosen label: " + str(label_n))
        # list_of_newly_labeled[new_index] = label_n

        unknown_indexes.remove(n)
        # print(n)

        title = "Iteration 0"
        name = folder + ("test_0b.jpg")
        # name = folder + ("rs2it20_0b.jpg")
        filenames.append(name)
        # plot_svm(clf, train_indexes, unknown_indexes, n, title, name)

        num = 5
        # num = 20
        t = []
        for i in range(num):
            train_indexes.append(n)
            y_pool.iloc[n] = int(label_n)
            X_train = X_pool.iloc[train_indexes]
            y_train = y_pool.iloc[train_indexes]
            print(y_train)
            clf = LinearSVC()
            clf.fit(X_train, y_train.to_list())
            title, name = "Iteration " + str(i + 1), folder + ("test_%d.jpg" % (i + 1))
            # title, name = "Iteration "+str(i+1), folder + ("rs2it20_%d.jpg" % (i+1))

            n = find_most_ambiguous(clf, unknown_indexes)

            print(X_pool.iloc[n])
            label_n = input(
                "What label would you give this datapoint (also shown by the star)? \n 0 for not interesting, 1 for interesting. \n")
            print("you have chosen label: " + str(label_n))

            unknown_indexes.remove(n)
            plot_svm(clf, train_indexes, unknown_indexes, n, title, name)
            filenames.append(name)




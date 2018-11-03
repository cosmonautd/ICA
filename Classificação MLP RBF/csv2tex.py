import numpy

acc_arr = [0.975, 0.97, 0.97, 0.97333, 0.97333, 0.97667, 0.97, 0.97, 0.97, 0.965]
recall_arr = []
precision_arr = []

acc_classes = numpy.array(30)
recall_classes = numpy.array(30)
precision_classes = numpy.array(30)

for d in range(1,11):

    data = numpy.genfromtxt('confmatrix-mlp-200-total-%d.csv' % d, delimiter=',')

    # for i, row in enumerate(data):
    #     row = list(map(int,row))
    #     print('&%02d \\vrule & %s \\\\' % (i+1, ' & '.join(list(map(str,row)))))

    acc = [data[i][i]/20 for i in range(30)]
    recall = [data[i][i]/sum([data[i][j] for j in range(30)]) if sum([data[i][j] for j in range(30)]) != 0 else 1 for i in range(30)]
    precision = [data[i][i]/sum([data[j][i] for j in range(30)]) if sum([data[j][i] for j in range(30)]) != 0 else 1 for i in range(30)]

    recall_arr.append(numpy.mean(recall))
    precision_arr.append(numpy.mean(precision))

# print(numpy.mean(acc_arr))
# print(numpy.std(acc_arr))

# print(numpy.mean(recall_arr))
# print(numpy.std(recall_arr))

# print(numpy.mean(precision_arr))
# print(numpy.std(precision_arr))
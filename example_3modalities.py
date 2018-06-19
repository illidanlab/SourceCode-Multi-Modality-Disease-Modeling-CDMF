import tensorflow as tf
import numpy as np
import os
import scipy.io
import math
from sklearn.metrics import roc_auc_score


# define Frobenius norm square
def frob(z):
    vec_i = tf.reshape(z, [-1])
    return tf.reduce_sum(tf.mul(vec_i, vec_i))


def sigmoid(x):
    return 1 / (1 + tf.exp(-x))


def logistic_loss(w,b, label, x, sample_size):
    return tf.reduce_sum(tf.log(1.0 / sigmoid(tf.mul(label, tf.matmul(x, w)+b))))/sample_size


def train(max_steps, tol, xi_data, xg_data, xs_data, y_data, du1, du2, iter, train_size, lambda1, lambda2, lambda3, lambda4, lambda5, lambda6):
    ndti, di = xi_data.shape
    n, dg = xg_data.shape
    _, ds = xg_data.shape
    tf.set_random_seed(1)
    sess = tf.InteractiveSession()
    # Input placeholders
    with tf.name_scope("input"):
        xi = tf.placeholder(tf.float32, shape=(None, di), name='xi-input')
        xg = tf.placeholder(tf.float32, shape=(None, dg), name='xg-input')
        xs = tf.placeholder(tf.float32, shape=(None, ds), name='xs-input')
        y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')
        keep_prob = tf.placeholder(tf.float32)
    # initialize all factors by svd, you can choose other way to initialize all the variables
    with tf.name_scope('svd'):
        udti_svd1, _, vdti_svd1 = np.linalg.svd(xi_data, full_matrices=False)
        udti_svd2, _, vdti_svd2 = np.linalg.svd(udti_svd1, full_matrices=False)

        ut1_svd1, _, vt1_svd1 = np.linalg.svd(xg_data, full_matrices=False)
        ut1_svd2, _, vt1_svd2 = np.linalg.svd(ut1_svd1, full_matrices=False)

        us_svd1, _, vs_svd1 = np.linalg.svd(xs_data, full_matrices=False)
        us_svd2, _, vs_svd2 = np.linalg.svd(us_svd1, full_matrices=False)

        vi = tf.cast(tf.Variable(vdti_svd1[0: du1, :]), tf.float32)
        vg = tf.cast(tf.Variable(vt1_svd1[0: du1, :]), tf.float32)
        vs = tf.cast(tf.Variable(vs_svd1[0: du1, :]), tf.float32) 

        u1dti = udti_svd2[0: du2, 0: du1]
        u1t1 = ut1_svd2[0: du2, 0:du1]
        u1s = us_svd2[0: du2, 0:du1]
        u2_final = ut1_svd2[:, 0:du2]
        u2 = tf.cast(tf.Variable(u2_final), tf.float32)

    w = tf.get_variable("w", shape=(du2, 1), initializer=tf.contrib.layers.xavier_initializer())
    u1i = tf.Variable(tf.cast(u1dti, tf.float32))
    u1g = tf.Variable(tf.cast(u1t1, tf.float32))
    u1s = tf.Variable(tf.cast(u1s, tf.float32))
    bias = tf.Variable(tf.constant(0.1, shape=[1,1]))


    with tf.name_scope('output'):
        u_train = u2[0: train_size, :]
        u_test = u2[train_size: n, :]
        y_train = tf.sign(tf.matmul(u_train, w)+bias)
        y_test = tf.sign(tf.matmul(u_test, w)+bias)
        y_conf = tf.matmul(u_test, w) + bias
        train_conf = tf.matmul(u_train, w) + bias 
  
    loss = logistic_loss(w, bias, y_, u_train, train_size) + \
           frob(xi - tf.matmul(u2,tf.square(tf.matmul(u1i,vi)))) + \
           frob(xg - tf.matmul(u2,tf.square(tf.matmul(u1g, vg)))) + \
           frob(xs - tf.matmul(u2, tf.square(tf.matmul(u1s, vs))))+\
           lambda1*frob(u1i) + lambda1*frob(u1g) + lambda2*frob(u1s)\
           + lambda3*frob(u2) + lambda4*frob(vi) + lambda5*frob(vg) + lambda6*frob(vs)

    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    tf.global_variables_initializer().run()

    def feed_dict(training, prob_rate):
        xis = xi_data
        xgs = xg_data
        xss = xs_data
        if training:
            ys = y_data[0:train_size, :]
        else:
            ys = y_data[train_size: n, :]
        return {xi: xis, xg: xgs, xs:xss, y_: ys, keep_prob: prob_rate}

    funval = []
    _, loss_iter = sess.run([train_step, loss], feed_dict=feed_dict(True, 0.8))
    funval.append(loss_iter)
    for i in range(max_steps):
        _, loss_iter = sess.run([train_step, loss], feed_dict=feed_dict(True, 0.8))
        funval.append(loss_iter)
        if abs(funval[i+1] - funval[i]) < tol:
            break
        if math.isnan(loss_iter):
            break
    pred_test, pred_conf = sess.run([y_test, y_conf], feed_dict=feed_dict(False, 1.0))
    pred_test_reshape = np.reshape(np.array(pred_test), (n-train_size, 1))
    pred_conf_reshape = np.reshape(np.array(pred_conf), (n-train_size, 1))
    label_test = y_data[train_size: n, :]
    test_auc = roc_auc_score(label_test, pred_conf_reshape)
    pred_train, train_conf = sess.run([y_train,train_conf], feed_dict=feed_dict(True, 1.0))
    pred_train_reshape = np.reshape(np.array(pred_train), (train_size, 1))
    train_conf_reshape = np.reshape(np.array(train_conf), (train_size, 1))
    label_train = y_data[0:train_size, :]
    train_auc = roc_auc_score(label_train, train_conf_reshape)
    w, u1i, u1g, u1s, bias, u2, vi, vg = sess.run([w,u1i,u1g,u1s,bias,u2,vi,vg], feed_dict=feed_dict(True, 1.0))
    sess.close()
    return {'funval': funval, 'y_train_conf': train_conf_reshape, 'y_test_conf':pred_conf_reshape,'train_auc': train_auc, 'test_auc': test_auc, 'y_pred': pred_test_reshape, 'y_label':label_test, 'w': w, 'u1i': u1i, 'u1g': u1g, 'u1s':u1s, 'bias': bias, 'u2':u2, 'vi':vi, 'vg':vg}


def load_data(iter):
    # please load modality 1 as x1, modality 2 as x2 and modality 3 as x3, label as y, also please indicate training sample size
    x1 = None
    x2 = None
    x3 = None
    y = None
    train_size = None
    return {'x1': x1, 'x2': x2,'x3':x3, 'y': y, 'train_size': train_size}

def main():
    max_iter = 200000
    tol = 1e-7
    directory = "result"
    if not os.path.exists(directory):
        os.makedirs(directory)
    # du1 is the nodes number of first hidden layer
    for du1 in [50, 100, 150]:
        directory1 = directory + "/du1" + str(du1)
        if not os.path.exists(directory1):
            os.makedirs(directory1)
        # du2 is the nodes number of second hidden layer
        for du2 in [30, 50, 70, 110, 130]:#90
            directory2 = directory1 + "/du2" + str(du2)
            if not os.path.exists(directory2):
                os.makedirs(directory2)
            # auc.txt is used to save all auc
            aucfile = open(directory2 + "/auc.txt", 'w')
            # tune regularization parameters
            for lambdax in[1e-4, 1e-3, 1e-2]:
                lambda1 =1e-2
                lambda2 =1e-2
                lambda3 =lambdax 
                lambda4 =1e-2
                lambda5 =1e-2
                lambda6 =1e-2
                auc = []
                train_auc = []
                # iteration number
                for it in [1, 2, 3, 4, 5]:
                    data = load_data(it)
                    x1 = data['x1']
                    x2 = data['x2']
                    x3 = data['x3']
                    y = data['y']
                    train_size = data['train_size']
                    # train the network
                    result = train(max_iter, tol, x1, x2, x3, y,du1, du2, it, train_size,lambda1, lambda2, lambda3, lambda4, lambda5, lambda6)
                    auc.append(result['test_auc'])
                    print "iter" + str(it) + "train" + str(result['train_auc']) + "test"+ str(result['test_auc']) 
                    train_auc.append(result['train_auc'])
                auc_mean = np.mean(auc)
                train_auc_mean = np.mean(train_auc)
                auc_std = np.std(auc)
                print "lambdax" +str(lambdax) + ":train_auc" + str(train_auc_mean) + "test_auc" + str(auc_mean)
                aucfile.write("lambda1 %slambda2 %s lambda3 %s lambda4 %s lambda5 %s:train_auc %s auc_mean %s auc_std %s\n"
                              % (lambda1,lambda2, lambda3, lambda4, lambda5,train_auc_mean, auc_mean, auc_std))
            aucfile.close()
if __name__ == "__main__":
    main()


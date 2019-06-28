import numpy as np








def call_train_core(n,samples,k,workers_train_label_use_core,workers_val_label,fname,epochs,depth,gpus):
    # this function takes as input the same variables as the "call_train" function and it calls
    # the mxnet implementation of ResNet training module function "train"
    workers_train_label = {}
    workers_train_label['softmax0_label'] = workers_train_label_use_core
    prediction, val_acc = train(gpus,fname,workers_train_label,workers_val_label,numepoch=epochs,batch_size=500,depth = depth,lr=0.5)
    model_pred = np.zeros((n,k))
    model_pred[np.arange(samples), np.argmax(prediction[0:samples],1)] = 1
    return model_pred, val_acc


def train(gpus, fname, workers_train_label, workers_val_label, numepoch, batch_size, depth=20, lr=0.5):
    output_filename = "tr_err.txt"
    model_num = 1
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if os.path.isfile(output_filename):
        os.remove(output_filename)
    hdlr = logging.FileHandler(output_filename)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)

    kv = mx.kvstore.create('device')
    ### training iterator
    train1 = mx.io.ImageRecordIter(
        path_imgrec=fname[0],
        label_width=1,
        data_name='data',
        label_name='softmax0_label',
        data_shape=(3, 32, 32),
        batch_size=batch_size,
        pad=4,
        fill_value=127,
        rand_crop=True,
        max_random_scale=1.0,
        min_random_scale=1.0,
        rand_mirror=True,
        shuffle=False,
        num_parts=kv.num_workers,
        part_index=kv.rank)

    ### Validation iterator
    val1 = mx.io.ImageRecordIter(
        path_imgrec=fname[2],
        label_width=1,
        data_name='data',
        label_name='softmax0_label',
        batch_size=batch_size,
        data_shape=(3, 32, 32),
        rand_crop=False,
        rand_mirror=False,
        pad=0,
        num_parts=kv.num_workers,
        part_index=kv.rank)

    n = workers_train_label['softmax0_label'].shape[0]
    k = workers_train_label['softmax0_label'].shape[1]
    n1 = workers_val_label['softmax0_label'].shape[0]
    train2 = mx.io.NDArrayIter(np.zeros(n), workers_train_label, batch_size, shuffle=False, )
    train_iter = MultiIter([train1, train2])
    val2 = mx.io.NDArrayIter(np.zeros(n1), workers_val_label, batch_size=batch_size, shuffle=False, )
    val_iter = MultiIter([val1, val2])

    if ((depth - 2) % 6 == 0 and depth < 164):
        per_unit = [int((depth - 2) / 6)]
        filter_list = [16, 16, 32, 64]
        bottle_neck = False
    else:
        raise ValueError("no experiments done on detph {}, you can do it youself".format(depth))
    units = per_unit * 3
    symbol = resnet(units=units, num_stage=3, filter_list=filter_list, num_class=k, data_type="cifar10",
                    bottle_neck=False, bn_mom=0.9, workspace=512,
                    memonger=False)

    devs = mx.cpu() if gpus is None else [mx.gpu(int(i)) for i in gpus.split(',')]
    epoch_size = max(int(n / batch_size / kv.num_workers), 1)
    if not os.path.exists("./model" + str(model_num)):
        os.mkdir("./model" + str(model_num))
    model_prefix = "model" + str(model_num) + "/resnet-{}-{}-{}".format("cifar10", depth, kv.rank)
    checkpoint = mx.callback.do_checkpoint(model_prefix)

    def custom_metric(label, softmax):
        return len(np.where(np.argmax(softmax, 1) == np.argmax(label, 1))[0]) / float(label.shape[0])

    # there is only one softmax layer with respect to which error of all the labels are computed
    output_names = []
    output_names = output_names + ['softmax' + str(0) + '_output']
    eval_metrics = mx.metric.CustomMetric(custom_metric, name='accuracy', output_names=output_names,
                                          label_names=workers_train_label.keys())

    model = mx.mod.Module(
        context=devs,
        symbol=mx.sym.Group(symbol),
        data_names=['data'],
        label_names=workers_train_label.keys(),  # ['softmax0_label']
    )
    lr_scheduler = multi_factor_scheduler(0, epoch_size, step=[40, 50], factor=0.1)
    optimizer_params = {
        'learning_rate': lr,
        'momentum': 0.9,
        'wd': 0.0001,
        'lr_scheduler': lr_scheduler}

    model.fit(
        train_iter,
        eval_data=val_iter,
        eval_metric=eval_metrics,
        kvstore=kv,
        batch_end_callback=mx.callback.Speedometer(batch_size, 50),
        epoch_end_callback=checkpoint,
        optimizer='nag',
        optimizer_params=optimizer_params,
        num_epoch=numepoch,
        initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2),
    )

    epoch_max_val_acc, train_acc, val_acc = max_val_epoch(output_filename)
    # print "val-acc: " + str(val_acc)

    # Prediction on Training data
    sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, epoch_max_val_acc)
    model = mx.mod.Module(
        context=devs,
        symbol=sym,
        data_names=['data'],
        label_names=workers_train_label.keys(),  # ['softmax0_label']
    )
    model.bind(for_training=False, data_shapes=train_iter.provide_data,
               label_shapes=train_iter.provide_label, )
    model.set_params(arg_params, aux_params, allow_missing=True)

    outputs = model.predict(train_iter)
    if type(outputs) is list:
        return outputs[0].asnumpy(), val_acc
    else:
        return outputs.asnumpy(), val_acc
function N_acc = ACC19(aa, bb, cc, train, test, order)

    tr3d = train(:,3:end);  te3d = test(:,3:end);  N_acc = zeros(1,19);
    train_label = train(:,1); test_label = test(:,1); d = size(train(:,3:end),2);
    if size(order,1) > size(order,2)
        order = order.';
    end

    trainn = zeros(size(tr3d));  testt = zeros(size(te3d));
    sele_feature_N = round((aa: cc: bb)*d);
    for sele_N = 1: 1: length(sele_feature_N)
        sele_feature = order(1,1:sele_feature_N(sele_N));
        trainn(:,sele_feature) = tr3d(:,sele_feature);
        testt(:,sele_feature) = te3d(:,sele_feature);

        cmd = ['-s 0 -t 0 -c 1 -q'];
        model = svmtrain(train_label, trainn, cmd);
        [predict_label, accuracy, ~] = svmpredict(test_label, testt, model);
        acc = accuracy(1,1)/100*length(predict_label);
        N_acc(1, sele_N) = acc;
        clear trainn testt sele_feature cmd model predict_label accuracy acc
    end

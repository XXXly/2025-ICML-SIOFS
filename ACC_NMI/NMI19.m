function nmi = NMI19(aa, bb, cc, train, feat_idx)

	nmi = zeros(1,19);   d = size(train(:,3:end),2);
	train_inst = train(:,3:end);   train_label = train(:,1);
    c = length(unique(train_label));
	train_inst = NormalizeFea(train_inst);
    sele_feature_N = round((aa: cc: bb)*d);
    for sele_N = 1: 1: length(sele_feature_N)
        sele_feature = feat_idx(1,1:sele_feature_N(sele_N));
        selected_inst = train_inst(:,sele_feature);
        res = litekmeans_NoRan(selected_inst,c);
        res = bestMap(train_label,res);
        nmi(1,sele_N) = nmi(1,sele_N) + MutualInfoo(train_label,res);
        clear sele_feature selected_inst res
    end


clear; close all;
addpath ./ACC_NMI
addpath ./ACC_NMI/libsvm-3.23
path = './data/';
% Input: train, test. The rows of "train" and "test" matrices are the instances.
% In "train" and "test", both first columns are labels;
KK = 5;
aa = 50;  bb = 300;  cc = 50;   % the numbers of selected top features,50,100,бн,300
cal_N = length(aa:cc:bb);
result_ACC = cell(1,1);  result_NMI = cell(1,1);
for alpha = round(0.05: 0.05: 0.95, 3)

    SIO_acc = zeros(1,cal_N);  SIO_nmi = zeros(1,cal_N);  N_test = 0;
    for ii = 1: 1: KK
        load(strcat(path,'Lung_dis_traintest-',num2str(ii),'.mat'));
        train_label = train(:,1);
        train_inst = train(:,3:end);
        C0 = length(unique([train_label;test(:,1)]));
        [~, feature_index] = SIOFS(train_inst, train_label, alpha, C0);
 
        if isempty(feature_index)
            return
        end

        SIO_acc(1,:) = SIO_acc(1,:) + ACC6(aa,bb,cc,train,test,feature_index);
        SIO_nmi(1,:) = SIO_nmi(1,:) + NMI6(aa, bb, cc, train, feature_index);
        N_test = N_test + size(test,1);
        clear train_label train_inst C0 feature_index
    end

    SIO_acc = SIO_acc./N_test*100;
    resu_acc = [round([mean(SIO_acc(1,:)) std(SIO_acc(1,:),1) SIO_acc(1,:)],2) alpha];
    result_ACC{1,1} = [result_ACC{1,1}; resu_acc];
    SIO_nmi = SIO_nmi/KK;
	resu_nmi = [round([mean(SIO_nmi(1,:)) std(SIO_nmi(1,:),1) SIO_nmi(1,:)],4) alpha];
    result_NMI{1,1} = [result_NMI{1,1}; resu_nmi];
    clear resu_acc resu_nmi SIO_acc SIO_nmi
end  %end for alpha
result_ACC = result_ACC{1,1};  result_NMI = result_NMI{1,1};
function [feature_score, feature_index] = SIOFS(train_inst, train_label, alpha, C0)
% The code was written by Lixin Yuan (Lixin Yuan, yuanlixin@hhu.edu)
%   originally version -- 2025.5.19
%
% Originally I thought this is a trivial function, and did not put it online 
%   Recently, (after giving talks & receiving emails), I realized that some details 
% are still important.
%   Although the algorithm is simple, a wrong implementation could still happen.
%   Thus, comments are added and this file is put online. 

% Please kindly cite the paper at
% @inproceedings{
% anonymous2025stray,
% title={Stray Intrusive Outliers-Based Feature Selection on Intra-Class Asymmetric Instance Distribution or Multiple High-Density Clusters},
% author={Lixin Yuan, Yirui Wu, Wenxiao Zhang, Minglei Yuan, Jun Liu},
% booktitle={Forty-second International Conference on Machine Learning},
% year={2025},
% url={https://openreview.net/forum?id=kUagmiIN5x}
% }

	classes = unique(train_label);
	c = length(classes);
	d = size(train_inst,2);
	Center = zeros(c,d);
	data_k = cell(c, 1);
    for k = 1: 1: c
        [idx_k, ~] = find(train_label == classes(k));
        data_k{k,1} = train_inst(idx_k, :);
        Center(k,:) = RDM_center(data_k{k,1}, alpha);
        clear idx_k
    end
	feature_score = intrusion_degree(c, data_k, Center, alpha, C0);

% All instances have the same feature value. This means this feature has
% no inter-class distinctiveness, and is a low-quality feature.
	all_std = std(train_inst,1,1);
	[~,std0] = find(all_std == 0);
    if ~isempty(std0)
        feature_score(:,std0) = inf;
    end
	[~, feature_index] = sort(feature_score);
end

function RDM_center = RDM_center(data_class, alpha)

    Nk = size(data_class,1);
    if Nk >= 3
        dist_matrix = pdist2(data_class,data_class,'minkowski',1);
        row_median = median(dist_matrix,2);

        idx_density = row_median <= max(mink(row_median,ceil(alpha*Nk)));
        RDM_center = mean(data_class(idx_density,:),1);
    else
	RDM_center = mean(data_class,1);
    end
end

function SIO_Rela = intrusion_degree(c, Fk_ori, center, alpha, C0)

    R_far = zeros(c,1);
    kL_CC_vec = cell(c,c);
    intrusion_mean = cell(1,1);
    kk_vec = cell(c,c);
    kL_vec = cell(c,c);

% Computing the distance between pairwise class centers and
% the radius (жи^(l)) of class majority
    for k = 1: 1: c
        for L = k-1 : -1 : 1
            kL_CC_vec{k,L} = abs(bsxfun(@minus, center(k,:), center(L,:)));
            kL_CC_vec{L,k} = kL_CC_vec{k,L};
        end
        kk_iC_vec = bsxfun(@minus, Fk_ori{k,1}, center(k,:));
        dist_iC = sum(abs(kk_iC_vec),2);

        d_cen = RDM_center(dist_iC, alpha);
        devi = bsxfun(@minus, dist_iC, d_cen);
        d_std = sqrt(mean(devi.^2));
        if d_std ~= 0
            CS = sum(devi.^3)/(size(dist_iC,1)*d_std^3);
        else
            CS = 0;
        end

        R_far(k,1) = d_cen + (2 - CS/3)*d_std;
        clear kk_iC_vec dist_iC d_cen devi d_std CS
    end
    if min(R_far) == 0
        RR_far = R_far;
        RR_far(RR_far==0) = [];
        R_far(R_far==0) = std(RR_far,1)/mean(RR_far)*(C0-1)*min(RR_far);
    end

 % Recognizing potential SIO of class $k$ towards class $l$
    for k = 1: 1: c
        for i= 1: 1: size(Fk_ori{k,1}, 1)
            kL_iC_vec = abs(bsxfun(@minus, Fk_ori{k,1}(i,:),center));
            dist_kL_iC = sum(kL_iC_vec,2);
            dist_kL_iC(k) = inf;
            [idx_in,~] = find(dist_kL_iC < R_far);
            if ~isempty(idx_in)
                if length(idx_in) > 1
                    [Rk,~] = find(dist_kL_iC(idx_in,:) == min(dist_kL_iC(idx_in,:)));
                    idx_SIO = idx_in(Rk);
                else
                    idx_SIO = idx_in;
                end
                kk_vec{k,idx_SIO(1)} = [kk_vec{k,idx_SIO(1)}; kL_iC_vec(k,:)];

                for kL_N = 1: 1: length(idx_in)
                    kL_vec{k,idx_in(kL_N)} = [kL_vec{k,idx_in(kL_N)}; kL_iC_vec(k,:)];
                end
            end
            clear kL_iC_vec dist_kL_iC idx_in Rk idx_SIO kL_N
        end
    end

% Examining whether $\mathbf{x}_i^{(k)}\in \mathcal{X}^{(kl_0)}$ is the SIO
% from class $k$ towards class $l_0$ if $\mathcal{X}^{(kl_0)} \neq \emptyset$
    for k = 1: 1: c
        for L = 1:1:c
            if ~isempty(kk_vec{k, L})
                if ~isempty(kL_vec{L,k})
                    RkRL = bsxfun(@plus, kk_vec{k, L}, mean(kL_vec{L,k},1));
                    intrusion_vec = bsxfun(@minus, RkRL, kL_CC_vec{k,L});
                    [idx_Re,~] = find(sum(intrusion_vec,2) > 0);
                    if ~isempty(idx_Re)
                        intrusion_mean{1,1} = [intrusion_mean{1,1}; mean(intrusion_vec(idx_Re,:),1)];
                    end
                    clear RkRL intrusion_vec idx_Re
                end
            end
        end
    end

    intrusion_degree = sum(intrusion_mean{1,1},2);
    SIO_sele = ceil(0.5*size(intrusion_degree,1));
    [idx_over,~] =  find(intrusion_degree <= max(mink(intrusion_degree,SIO_sele)));
    SIO_Rela = mean(intrusion_mean{1,1}(idx_over,:),1);
end
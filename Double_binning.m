%%%-------------------------找最大樣本量bin二次分bin,評判有效bin後整合其他bin
clc;
clear;
close all;

num_datasets = 9;  % 數據集數量
num_iterations = 10;  % 每個數據集的迭代次數

dataset_names = { ...
    'MalMem2022_1K', ...
    'mailspam', ...
    'website_phishing', ...
    'phishing_libsvm', ...
    'Phishing_Legitimate', ...
    'PhiUSIIL_Phishing_URL', ...
    'NSLKDD_1K', ...
    'android', ...
    'creditcard_2023' ...
};

accuracy_results = zeros(num_iterations, num_datasets); 
time_results = zeros(1, num_datasets); % 每個數據集的平均計算時間

tic;
for k = 9
    
    switch k
        case 1
            data_path = 'hash_MalMem2022_1K_reduced_data.csv';
            filename = 'MalMem2022_1K';
        case 2
            data_path = 'mailspam_1k.csv';
            filename = 'mailspam';
        case 3
            data_path = 'website_phishing_1k.csv';
            filename = 'website_phishing';
        case 4
            data_path = 'phishing_libsvm_1k.csv';
            filename = 'phishing_libsvm';
        case 5
            data_path = 'Phishing_Legitimate_full_1k.csv';
            filename = 'Phishing_Legitimate';
        case 6
            data_path = 'PhiUSIIL_Phishing_URL_Dataset_1k.csv';
            filename = 'PhiUSIIL_Phishing_URL';
        case 7 
            data_path = 'NSLKDD_Train_1K.csv';
            filename = 'NSLKDD_1K';
        case 8
            data_path = 'android_1k.csv';
            filename = 'android';    
        case 9
            data_path = 'creditcard_2023_1k.csv';
            filename = 'creditcard_2023';
    end
    filename = dataset_names{k};

    accuracy_all = zeros(1, num_iterations);
    disp(['處理數據集: ', filename]);
    
    % 开始计时
    dataset_start_time = tic;
    
    for j = 1:num_iterations
        % 讀取和預處理數據
        input_data = readtable(data_path);
        disp('數據加載成功');
        input_data = normalize_func(input_data);
        [input_row, input_column] = size(input_data);
        input_data_cell = table2array(input_data(:, 1:input_column-1));
        variances = var(input_data_cell);
        input_data_cell = input_data_cell(:, variances > 0);
        meas = input_data_cell;
        labels = table2array(input_data(:, input_column));
        
        % 数据分割
        train_ratio = 0.7;
        num_samples = size(meas, 1);
        num_train = round(train_ratio * num_samples);
        train_indices = randperm(num_samples, num_train);
        test_indices = setdiff(1:num_samples, train_indices);
        train_data = meas(train_indices, :);
        train_labels = labels(train_indices);
        test_data = meas(test_indices, :);
        test_labels = labels(test_indices);

        %% 計算斯皮爾曼秩相關係數並排序特徵
        %[R, ~] = corr(train_data, train_labels, 'Type', 'Spearman'); % 計算斯皮爾曼相關係數
        %[~, featureIdx] = sort(abs(R), 'descend'); % 按相關係數的絕對值降序排序

        % 去掉倒數10個特徵
        %selectedFeaturesSpearman = featureIdx(1:end-24); % 移除倒數10個特徵

        % 選擇排序後的特徵
        %train_data = train_data(:, selectedFeaturesSpearman);
        %test_data = test_data(:, selectedFeaturesSpearman);

        % 特徵選擇
              [~, chi2values] = fscchi2(train_data, train_labels);
              [~, featureIdx] = sort(chi2values, 'descend');
              selectedFeaturesChi = featureIdx(1:16);
              train_data = train_data(:, selectedFeaturesChi);
              test_data = test_data(:, selectedFeaturesChi);


        % 初始化模型参数
        max_depth = 3;  
        lambda = 1;
        gamma = 0.1;
        n_estimators = 20;
        initial_learning_rate = 0.35;
        s_values = 0.001; 
        learning_rate = initial_learning_rate;
        bin_count = 256;

        % 训练 XGBoost 模型
        model = train_xgboosts(train_data, train_labels, max_depth, n_estimators, learning_rate, lambda, gamma, s_values, bin_count);

        % 使用测试数据进行预测
        pred = zeros(size(test_data, 1), 1);
        for t = 1:n_estimators
            learning_rate = initial_learning_rate * exp(-s_values * t);
            pred = pred + learning_rate * tree_predict(model.trees{t}, test_data);
        end

        prob = 1 ./ (1 + exp(-pred));
        prob(prob >= 0.5) = 1;
        prob(prob < 0.5) = 0;
        
        % 计算准确度
        accuracy = mean(prob == test_labels);
        accuracy_all(j) = accuracy;

        disp(['第', num2str(j), '次迭代准确度: ', num2str(accuracy)]);
    end
    
    accuracy_results(:, k) = accuracy_all;
    time_results(k) = toc(dataset_start_time) / num_iterations; % 每次迭代的平均时间
    disp(['数据集 ', filename,  ' 平均准确度: ', num2str(mean(accuracy_all))]);

    % 显示数据集的总时间
    elapsed_time = toc(dataset_start_time);
    disp(['处理数据集 ', filename,  ' 的总时间: ', num2str(elapsed_time), ' 秒']);
end

% 显示所有数据集的平均准确度和平均計算時間
disp('所有数据集的平均准确度和平均计算时间:');
for k = 1:num_datasets
    name = dataset_names{k};
    fprintf('数据集 %s 平均准确度: %.4f, 平均时间: %.4f 秒\n', name, mean(accuracy_results(:, k)), time_results(k));
end

end_time = toc;
disp('数据集總时间:');
disp(end_time);



function [best_feature, best_threshold] = find_best_split(X, grad, hess, lambda, gamma, bin_count)
    best_gain      = -inf;
    best_feature   = -1;
    best_threshold = -1;
    [~, num_features] = size(X);

    for feature_index = 1:num_features
        vals = X(:, feature_index);

        % —— 跳过全 0 特征 —— %
        if all(isnan(vals))
            continue;
        end

        % —— 第一次 256 bin 粗分 —— %
        [counts256, edges256] = histcounts(vals, bin_count);
        % dynamic_thresh = mean(counts256);
        k=2;
        mu  = mean(counts256);                 % 均值
        d   = abs(counts256 - mu);             % 到均值的绝对距离
        MAD = mean(d);                 % 平均绝对偏差
        dynamic_thresh = mu + k * MAD;

        % —— 找出所有密集箱并分段 —— %
        dense_idx = find(counts256 >= dynamic_thresh);
        if isempty(dense_idx)
            new_edges = edges256;
        else
            % 分组连续区间
            segments = [];
            start_i = dense_idx(1);
            for k2 = 2:length(dense_idx)
                if dense_idx(k2) ~= dense_idx(k2-1) + 1
                    segments(end+1,:) = [start_i, dense_idx(k2-1)]; %#ok<AGROW>
                    start_i = dense_idx(k2);
                end
            end
            segments(end+1,:) = [start_i, dense_idx(end)];

            % 构造 new_edges
            new_edges = edges256(1);
            last_edge = 1;
            for si = 1:size(segments,1)
                s = segments(si,1);
                e = segments(si,2);
                % 合并稀疏区
                if s > last_edge
                    new_edges(end+1) = edges256(s); %#ok<AGROW>
                end
                % 保留密集区细分
                new_edges = [ new_edges, edges256(s+1:e+1) ]; %#ok<AGROW>
                last_edge = e + 1;
            end
            % 合并末尾稀疏区
            if last_edge < numel(edges256)
                new_edges(end+1) = edges256(end); %#ok<AGROW>
            end
            new_edges = unique(new_edges);
        end

        % —— 第二次按 new_edges 分箱累积 grad/hess —— %
        nbins = numel(new_edges) - 1;
        bin_idx  = discretize(vals, new_edges);
        bin_grad = accumarray(bin_idx, grad, [nbins, 1]);
        bin_hess = accumarray(bin_idx, hess, [nbins, 1]);

        % —— 增益计算 —— %
        G = sum(bin_grad); H = sum(bin_hess);
        G_left = 0; H_left = 0;
        for b = 1:nbins-1
            G_left  = G_left  + bin_grad(b);
            H_left  = H_left  + bin_hess(b);
            G_right = G - G_left;
            H_right = H - H_left;
            if H_left==0 || H_right==0, continue; end

            gain = (G_left^2/(H_left+lambda) + ...
                    G_right^2/(H_right+lambda) - ...
                    G^2/(H+lambda)) - gamma;

            if gain > best_gain
                best_gain      = gain;
                best_feature   = feature_index;
                best_threshold = new_edges(b+1);
            end
        end
    end
end


function node = build_tree(X, grad, hess, depth, max_depth, lambda, gamma, bin_count)
    % 初始化節點結構
    node = struct('feature_index', [], 'threshold', [], 'left', [], 'right', [], 'is_leaf', false, 'prediction', []);
    
    % 如果達到最大深度或梯度的唯一值數量為1（即無法繼續分裂），創建葉節點
    if depth >= max_depth || numel(unique(grad)) == 1
        node.is_leaf = true;
        node.prediction = -sum(grad) / (sum(hess) + lambda);
        return;
    end
    
    % 查找最佳分割點（以 bin 為基礎進行分裂）
    [feature_index, threshold] = find_best_split(X, grad, hess, lambda, gamma, bin_count);
    
    % 如果沒有找到有效的分割點，創建葉節點
    if feature_index == -1
        node.is_leaf = true;
        node.prediction = -sum(grad) / (sum(hess) + lambda);
        return;
    end
    
    % 根據最佳分割點將數據分為左右子集
    left_indices = X(:, feature_index) <= threshold;
    right_indices = ~left_indices;
    
    % 設置當前節點的分割特徵和閾值
    node.feature_index = feature_index;
    node.threshold = threshold;
    
    % 遞歸構建左子樹
    node.left = build_tree(X(left_indices, :), grad(left_indices), hess(left_indices), depth + 1, max_depth, lambda, gamma, bin_count);
    
    % 遞歸構建右子樹
    node.right = build_tree(X(right_indices, :), grad(right_indices), hess(right_indices), depth + 1, max_depth, lambda, gamma, bin_count);
end

function model = train_xgboosts(X, y, max_depth, num_trees, learning_rate, lambda, gamma, s_values, bin_count)
    model.trees = cell(num_trees, 1);
    model.learning_rate = learning_rate;
    model.lambda = lambda;
    model.gamma = gamma;
    model.s_values = s_values;
    model.bin_count = bin_count;

    % 計算標籤為1的比例
    pos_rate = mean(y);

    % 使用這個初始值來初始化預測
    pred = pos_rate * ones(size(y));
    
    for t = 1:num_trees
        learning_rate = learning_rate * exp(-s_values * t);
        
        % 計算預測的概率
        p = 1 ./ (1 + exp(-pred));
        
        % 計算梯度
        grad = p - y;
        
        % 計算 Hessian
        hess = p .* (1 - p);
        
        % 使用計算的梯度和 Hessian 构建树
        model.trees{t} = build_tree(X, grad, hess, 0, max_depth, lambda, gamma, bin_count);
        
        % 使用新树更新預測
        pred = pred + learning_rate * tree_predict(model.trees{t}, X);
    end
end

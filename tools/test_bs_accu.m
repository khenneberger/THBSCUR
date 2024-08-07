function [accu, result] = test_bs_accu(band_set, Dataset, classifier_type)
accu = struct('OA', 0, 'MA', 0, 'Kappa', 0);
train_X = Dataset.train_X;
train_labels = double(Dataset.train_labels);
test_X = Dataset.test_X;
test_labels = double(Dataset.test_labels);
test_size = size(test_labels, 1);
[no_rows, no_cols, ~] = size(Dataset.A);
unique_labels = unique(test_labels);
C = size(unique_labels,1);
% warning('off');
bs_train_X = train_X(:, band_set);
bs_test_X = test_X(:, band_set);
switch(classifier_type)
    case 'SVM'
        model = svmtrain(train_labels, bs_train_X, Dataset.svm_para); %'-c 300 -t 2 -g 1 -q'
        [predict_labels, corrected_num, ~] = svmpredict(test_labels, bs_test_X, model, '-q');
        result = svmpredict((1:no_rows*no_cols)', Dataset.X(band_set, :)', model, '-q');

        accu.OA = corrected_num(1) / 100;
        cmat = confusionmat(test_labels, predict_labels);
        sum_accu = 0;
        for i = 1 : C
            sum_accu = sum_accu + cmat(i, i) / sum(cmat(i, :), 2);
        end
        accu.MA = sum_accu / C;
        Pe = 0;
        for i = 1 : C
            Pe = Pe + cmat(i, :) * cmat(:, i);
        end
        Pe = Pe / (test_size * test_size);
        accu.Kappa = (accu.OA - Pe) / (1 - Pe);
    case 'CART'
        tree = fitctree(bs_train_X, train_labels);
        predict_label = tree.predict(bs_test_X);
        accu.OA = length(find(predict_label == test_labels)) / length(test_labels);
        cmat = confusionmat(test_labels, predict_label);
        sum_accu = 0;
        for i = 1 : C
            sum_accu = sum_accu + cmat(i, i) / sum(cmat(i, :), 2);
        end
        accu.MA = sum_accu / C;
        Pe = 0;
        for i = 1 : C
            Pe = Pe + cmat(i, :) * cmat(:, i);
        end
        Pe = Pe / (test_size * test_size);
        accu.Kappa = (accu.OA - Pe) / (1 - Pe);

    case 'KNN'
        predict_label = knnclassify(bs_test_X, bs_train_X, train_labels, 3, 'euclidean');
        accu.OA = 0;
        result = 0; %place holder (result is only for SVM)
        cmat = confusionmat(test_labels, predict_label);
        for i = 1 : size(predict_label, 1)
            if predict_label(i) == test_labels(i)
                accu.OA = accu.OA + 1;
            end
        end
        accu.OA = accu.OA / size(predict_label, 1);
        sum_accu = 0;
        for i = 1 : C
            sum_accu = sum_accu + cmat(i, i) / sum(cmat(i, :), 2);
        end
        accu.MA = sum_accu / C;

        Pe = 0;
        for i = 1 : C
            Pe = Pe + cmat(i, :) * cmat(:, i);
        end
        Pe = Pe / (test_size*test_size);
        accu.Kappa = (accu.OA - Pe) / (1 - Pe);
    case 'LDA'
        factor = fitcdiscr(bs_train_X, train_labels);
        predict_label = double(factor.predict(bs_test_X));
        cmat = confusionmat(test_labels, predict_label);
        accu.OA = length(find(predict_label == test_labels)) / length(test_labels);
        sum_accu = 0;
        for i = 1 : C
            sum_accu = sum_accu + cmat(i, i) / sum(cmat(i, :), 2);
        end
        accu.MA = sum_accu / C;

        Pe = 0;
        for i = 1 : C
            Pe = Pe + cmat(i, :) * cmat(:, i);
        end
        Pe = Pe / (test_size*test_size);
        accu.Kappa = (accu.OA - Pe) / (1 - Pe);
    case 'CNN1'
        % Define the CNN architecture here to ensure it's reset for each call
        
        % layers = [
        %     imageInputLayer([1, numFeatures, 1], 'Name', 'input', 'Normalization', 'none')
        %     fullyConnectedLayer(100, 'Name', 'fc1')
        %     reluLayer('Name', 'relu1')
        %     fullyConnectedLayer(C, 'Name', 'fc2')
        %     softmaxLayer('Name', 'softmax')
        %     classificationLayer('Name', 'output')];
        % 
        % options = trainingOptions('adam', ...
        %     'InitialLearnRate', 0.01, ...
        %     'MaxEpochs', 10, ...
        %     'MiniBatchSize', 16, ...
        %     'Shuffle', 'every-epoch', ...
        %     'Verbose', false);
        numFeatures = size(train_X, 2);  % Number of features based on band_set
        layers = [
            imageInputLayer([1, numFeatures, 1], 'Name', 'input')
            convolution2dLayer(3, 6, 'Padding', 'same')
            reluLayer
            batchNormalizationLayer
            maxPooling2dLayer(2, 'Stride', 2, 'Padding', 'same')
            convolution2dLayer(3, 16, 'Padding', 'same')
            reluLayer
            batchNormalizationLayer
            maxPooling2dLayer(2, 'Stride', 2, 'Padding', 'same')
            fullyConnectedLayer(120, 'Name', 'f1')
            reluLayer
            fullyConnectedLayer(84, 'Name', 'f2')
            reluLayer
            fullyConnectedLayer(C, 'Name', 'f3')  % Adjusted to output class size
            softmaxLayer
            classificationLayer
        ];
                
        options = trainingOptions('adam','MaxEpochs',25,'LearnRateSchedule' ,'piecewise','LearnRateDropPeriod',15,'LearnRateDropFactor' ,0.1,'Verbose', false);

        % Reshape the data to fit the CNN input requirements
        [numSamples, ~] = size(train_X);
        train_X_reshaped = reshape(train_X', [1, numFeatures, 1, numSamples]);
        train_Y = categorical(train_labels);

        % Train the CNN
        net = trainNetwork(train_X_reshaped, train_Y, layers, options);

        % Prepare and classify test data
        [numTestSamples, ~] = size(test_X);
        test_X_reshaped = reshape(test_X', [1, numFeatures, 1, numTestSamples]);
        predict_labels = classify(net, test_X_reshaped);

        % Calculate accuracy and other metrics
        accu.OA = sum(predict_labels == categorical(test_labels)) / test_size;
        cmat = confusionmat(categorical(test_labels), predict_labels);
        % Additional metrics calculation (MA, Kappa) can be added here

        % For CNN, 'result' can be detailed prediction results if needed
        result = predict_labels;  % or other relevant result metrics
        case 'CNN'
    % Extract features according to band_set
    bs_train_X = train_X(:, band_set);
    bs_test_X = test_X(:, band_set);
    [numSamples, ~] = size(bs_train_X);
    [numTestSamples, ~] = size(bs_test_X);
    % Number of features should be recalculated to reflect band_set
    numFeatures = size(bs_train_X, 2);  

    % Define the CNN architecture
    layers = [
        imageInputLayer([1, numFeatures, 1], 'Name', 'input', 'Normalization', 'none')
        convolution2dLayer(3, 6, 'Padding', 'same')
        reluLayer
        batchNormalizationLayer
        maxPooling2dLayer(2, 'Stride', 2, 'Padding', 'same')
        convolution2dLayer(3, 16, 'Padding', 'same')
        reluLayer
        batchNormalizationLayer
        maxPooling2dLayer(2, 'Stride', 2, 'Padding', 'same')
        fullyConnectedLayer(120, 'Name', 'f1')
        reluLayer
        fullyConnectedLayer(84, 'Name', 'f2')
        reluLayer
        fullyConnectedLayer(C, 'Name', 'f3')  % Adjusted to output class size
        softmaxLayer
        classificationLayer
    ];
                
    options = trainingOptions('adam', 'MaxEpochs', 25, 'LearnRateSchedule', 'piecewise', 'LearnRateDropPeriod', 15, 'LearnRateDropFactor', 0.1);

    % Reshape the training data to fit the CNN input requirements
    train_X_reshaped = reshape(bs_train_X', [1, numFeatures, 1, numSamples]);
    train_Y = categorical(train_labels);

    % Train the CNN
    net = trainNetwork(train_X_reshaped, train_Y, layers, options);

    % Prepare and classify test data
    test_X_reshaped = reshape(bs_test_X', [1, numFeatures, 1, numTestSamples]);
    predict_labels = classify(net, test_X_reshaped);

    % Calculate Overall Accuracy (OA)
    accu.OA = sum(predict_labels == categorical(test_labels)) / test_size;

    % Result output for CNN
    result = predict_labels;  % Or other relevant result metrics


end
end
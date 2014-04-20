trainData = loadMNISTImages('mnist/train-images-idx3-ubyte');
trainLabels = loadMNISTLabels('mnist/train-labels-idx1-ubyte');

trainLabels(trainLabels == 0) = 10;
acc = zeros(10, 1);

for i=1:10
    inputTrain = zeros(size(trainData, 1), size(trainData, 2)-6000);
    inputTest = zeros(size(trainData, 1), 6000);
    labelTrain = zeros(size(trainLabels, 1)-6000, 1);
    labelTest = zeros(6000, 1);
    k = 1;
    l = 1;
    for j=1:size(trainData, 2)
        if (j < (i-1)*6000+1 || j > i*6000)
            inputTrain(:,k) = trainData(:,j);
            labelTrain(k,:) = trainLabels(j,:);
            k = k+1;
        else
            inputTest(:,l) = trainData(:,j);
            labelTest(l,:) = trainLabels(j,:);
            l = l+1;
        end
    end
    size(inputTrain)
    size(inputTest)
    size(labelTrain)
    size(labelTest)
    acc(i,1) = stackedAE(inputTrain, labelTrain, inputTest, labelTest);
end

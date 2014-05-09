function [ cost, grad ] = stackedAECost(theta, inputSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda, data, labels)
                                         
% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example


%% Unroll softmaxTheta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

% You will need to compute the following gradients
stackgrad = cell(size(stack));
for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end

% You might find these variables useful
numCases = size(data, 2);
groundTruth = full(sparse(labels, 1:numCases, 1));


%% --------------------------- YOUR CODE HERE -----------------------------
%  Instructions: Compute the cost function and gradient vector for 
%                the stacked autoencoder.
%
%                You are given a stack variable which is a cell-array of
%                the weights and biases for every layer. In particular, you
%                can refer to the weights of Layer d, using stack{d}.w and
%                the biases using stack{d}.b . To get the total number of
%                layers, you can use numel(stack).
%
%                The last layer of the network is connected to the softmax
%                classification layer, softmaxTheta.
%
%                You should compute the gradients for the softmaxTheta,
%                storing that in softmaxThetaGrad. Similarly, you should
%                compute the gradients for each layer in the stack, storing
%                the gradients in stackgrad{d}.w and stackgrad{d}.b
%                Note that the size of the matrices in stackgrad should
%                match exactly that of the size of the matrices in stack.
%

input = cell(numel(stack)+1, 1);
delta = cell(numel(stack)+1, 1);
input{1} = data;

for d = 1:numel(stack)
    z = (stack{d}.w * input{d}) + repmat(stack{d}.b, 1, size(input{d}, 2));
    input{d+1} = sigmoid(z);
end

M = softmaxTheta * input{numel(stack)+1};

alpha = max(M);

M = M - repmat(alpha, numClasses, 1);

M = exp(M);

denom = sum(M);

M = M ./ repmat(denom, numClasses, 1);

logM = log(M);

costM = sum(sum(logM .* groundTruth));

cost = (costM / -numCases) + ((lambda / 2) * sum(sum(softmaxTheta .^ 2)));

delta{numel(stack)+1} = softmaxTheta' * (M - groundTruth);

softmaxThetaGrad = (((groundTruth - M) * input{numel(stack)+1}') ...
    ./ -numCases) + (lambda .* softmaxTheta);

for d = numel(stack):-1:1
    delta{d} = input{d} .* (ones(size(input{d})) - input{d}) .* (stack{d}.w' * delta{d+1});
    stackgrad{d}.w = ((delta{d+1} * input{d}') ./ size(input{d}, 2));
    stackgrad{d}.b = sum(delta{d+1}, 2) ./ size(input{d}, 2);
end

% -------------------------------------------------------------------------

%% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end

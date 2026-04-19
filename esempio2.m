% ESEMPIO 2
% Multi-class problem 
% Il file wine_dataset.mat (disponibile in Matlab) contiene il 
% dataset wineInputs  di 178 vini per ognuno dei quali vengono 
% raccolte tredici proprietà costituenti determinate a seguito 
% di un'analisi chimica:
%      1. Alcohol
%      2. Malic acid
%      3. Ash
%      4. Alcalinity of ash  
%      5. Magnesium
%      6. Total phenols
%      7. Flavanoids
%      8. Nonflavanoid phenols
%      9. Proanthocyanins
%     10. Color intensity
%     11. Hue
%     12. OD280/OD315 of diluted wines
%     13. Proline
% La variabile wineTargets è una matrice 3x178 le cui colonne  associano ciascun vino alla cantina produttrice:
% [1;0;0] --> cantina #1 
% [0;1;0] --> cantina #2
% [0;0;1] --> cantina #3


clear all
close all
clc
rng(2)
load wine_dataset;

% Standardization
wineInputs=(wineInputs - mean(wineInputs))./std(wineInputs);

% Dataset partition: 80% train set; 20% test set
[m,n]=size(wineInputs);
% partition the dataset into training set and test (or holdout) set 
P=cvpartition(n,'Holdout',0.2);
x_train=wineInputs(:,P.training);
y_train=wineTargets(:,P.training);
x_test=wineInputs(:,P.test);
y_test=wineTargets(:,P.test);

training_data=cell(2,P.TrainSize);
for k=1:P.TrainSize
    training_data{1,k}=x_train(:,k);
    training_data{2,k}=y_train(:,k);
end
test_data=cell(2,P.TestSize);
for k=1:P.TestSize
    test_data{1,k}=x_test(:,k);
    test_data{2,k}=y_test(:,k);
end

% configure and train the neural network
net=anetwork([m 15 25 3]);    
epochs=1e4;
mini_batch_size=1;
eta=1;
tic
net.sgd(training_data,epochs,mini_batch_size,eta,test_data)
fprintf('network trained in %g seconds \n',toc)

%% evaluate the net over the training set
y_pred_train=zeros(3,P.TrainSize);
for k=1:P.TrainSize
    y_pred=net.feedforward(training_data{1,k});
    % round to 0 or 1 according to the probability value
    [~,ind]=max(y_pred);
    y_pred_train(ind,k)=1;
end
% graphical output
figure(1)
clf
plotconfusion(y_train, y_pred_train);
set(gca,'FontSize',24)
title('Confusion matrix for the training set')

% evaluate the net over the test set
y_pred_test=zeros(3,P.TestSize);
for k=1:P.TestSize
    y_pred=net.feedforward(x_test(:,k));
    % round to 0 or 1 according to the probability value
    [~,ind]=max(y_pred);
    y_pred_test(ind,k)=1;
end
% graphical output
figure(2)
clf
plotconfusion(y_test, y_pred_test);
set(gca,'FontSize',24)
title('Confusion matrix for the test set')

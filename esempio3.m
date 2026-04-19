% ESEMPIO 3: problema di regressione (output_activation='linear)
%
% Regression problem
% 
clear all
close all
clc
rng(0)
[x_train, y_train] = simplefit_dataset;
% normalization
%x_train=(x_train - min(x_train))./(max(x_train)-min(x_train));
%y_train=(y_train - min(y_train))./(max(y_train)-min(y_train));


n=length(x_train);
training_data=cell(2,n);
for k=1:n
    training_data{1,k}=x_train(k);
    training_data{2,k}=y_train(k);
end
% configure and train the neural network
net=anetwork([1 30 20 20 20 1],'linear');     
epochs=1e4;
mini_batch_size=10;
eta=0.1;
tic
net.sgd(training_data,epochs,mini_batch_size,eta,training_data)
fprintf('network trained in %g seconds \n',toc)

%% evaluate the net over the training set
y_pred_train=zeros(1,n);
for k=1:n
    y_pred_train(k)=net.feedforward(training_data{1,k});
end
% graphical output
figure(1)
plot(x_train,y_train,'o',x_train,y_pred_train)
set(gca,'FontSize',24)
err=net.cost(training_data);
title(sprintf('MSE = %g',err))

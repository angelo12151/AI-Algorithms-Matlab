classdef anetwork < handle
    properties
        sizes
        num_layers
        weights
        biases
        output_activation 
        % 'sigmoid (default), 'linear' (regression problems)
        % 'softmax' (multiclass classification)
    end
    methods
        % Constructor
        function obj=anetwork(sizes,output_activation)
            if nargin<2
                output_activation='sigmoid';
            end

            obj.num_layers=length(sizes);
            obj.sizes=sizes;
            obj.weights=cell(obj.num_layers-1,1);
            obj.biases=cell(obj.num_layers-1,1);
            for k=1:obj.num_layers-1
                obj.weights{k}=randn(sizes(k+1),sizes(k));
                obj.biases{k}=randn(sizes(k+1),1);
            end 
            obj.output_activation=output_activation;
        end
        % Feedforward algorithm
        function a=feedforward(obj,a)
            for k=1:obj.num_layers-2
                a=sigma(obj.weights{k}*a+obj.biases{k});
            end
            k=obj.num_layers-1;
            a=sigma(obj.weights{k}*a+obj.biases{k},obj.output_activation);

        end
        % Stochastic gradient descent
        function sgd(obj,training_set,epochs,mini_batch_size,eta,test_set)
            test_set_provided=nargin==6;
            if test_set_provided
                [~,n_test]=size(test_set);
                cost_vector=zeros(epochs,1);
                figure(10)
                xlabel('epochs')
                ylabel('loss function')
                cost_vector(1)=obj.cost(test_set);
                cost_line=line(0,cost_vector(1));
                axis([0 epochs 0 cost_vector(1)])
                set(gca,'Fontsize',16)
            end


            [~,n]=size(training_set);
            for j=1:epochs
                ind=randperm(n);
                training_set=training_set(:,ind);
                for k=1:mini_batch_size:n
                    k1=min(k+mini_batch_size,n);
                    mini_batch=training_set(:,k:k1);
                    obj.update_mini_batch(mini_batch,eta);
                end
                if test_set_provided
                    cost_vector(j+1)=obj.cost(test_set);
                    if mod(j,fix(epochs/10))==0
                        figure(10)
                        set(cost_line,'XData',0:j,'YData',cost_vector(1:j+1))
                        drawnow
                    end
                    % basic learning rate change strategy
                    if cost_vector(j+1)>1.2*cost_vector(j)
                        eta=max(0.9*eta,1e-8);
                    elseif cost_vector(j+1)<0.8*cost_vector(j)
                        eta=min(1.1*eta,1e3);
                    end
                end
            end
        end
        % Loss function
        function C=cost(obj,test_set)
            % Mean Squared Error
            [~,n_test]=size(test_set);
            ypred=obj.feedforward(cell2mat(test_set(1,:)));
            C=0;
            for k=1:n_test
                C=C+norm(ypred(:,k)-test_set{2,k})^2/2;
            end
            C=C/n_test;
        end


    end
    methods (Access='private')
        % executes a step of stochastic gradient descent
        function update_mini_batch(obj,mini_batch,eta)
            [~,m]=size(mini_batch);
            etam=eta/m;
            % memory allocation 
            size_nabla_b=cellfun(@size,obj.biases,'UniformOutput',false);
            nabla_b=cellfun(@zeros,size_nabla_b,'UniformOutput',false);
            size_nabla_w=cellfun(@size,obj.weights,'UniformOutput',false);
            nabla_w=cellfun(@zeros,size_nabla_w,'UniformOutput',false);
            for j=1:m
                x=mini_batch{1,j};
                y=mini_batch{2,j};
                [delta_nabla_w,delta_nabla_b]=obj.backprop(x,y,etam);
                nabla_w=cellfun(@plus,nabla_w,delta_nabla_w,'UniformOutput',false);
                nabla_b=cellfun(@plus,nabla_b,delta_nabla_b,'UniformOutput',false);
            end
            obj.weights=cellfun(@minus,obj.weights,nabla_w,'UniformOutput',false);
            obj.biases=cellfun(@minus,obj.biases,nabla_b,'UniformOutput',false);
        end
        % back propagation algorithm
        function [nabla_w,nabla_b]=backprop(obj,x,y,etam)
            % memory allocation for nabla_b and nabla_w
            size_nabla_b=cellfun(@size,obj.biases,'UniformOutput',false);
            nabla_b=cellfun(@zeros,size_nabla_b,'UniformOutput',false);
            size_nabla_w=cellfun(@size,obj.weights,'UniformOutput',false);
            nabla_w=cellfun(@zeros,size_nabla_w,'UniformOutput',false);
            % memory allocation for z (weighted input vectors)
            % same shape as nabla_b
            z=nabla_b;
            % memory allocation for a (activation vectors)
            a=[{x};nabla_b];
            L=obj.num_layers;
            % feedforward algorithm
            for k=1:L-2
                z{k}=obj.weights{k}*a{k}+obj.biases{k};
                a{k+1}=sigma(z{k});
            end
            k=L-1;
            z{k}=obj.weights{k}*a{k}+obj.biases{k};
            a{k+1}=sigma(z{k},obj.output_activation);

            % backword propagation
            % last error vector
            delta=sigma_prime(z{L-1},obj.output_activation).*obj.cost_derivative(a{L},y);
            nabla_b{L-1}=etam*delta;
            nabla_w{L-1}=etam*delta*a{L-1}';
            for l=L-2:-1:1
                delta=sigma_prime(z{l}).*(obj.weights{l+1}'*delta);
                nabla_b{l}=etam*delta;
                nabla_w{l}=etam*delta*a{l}';
            end
        end
        % gradient of loss function w.r.t. aL
        function dCda=cost_derivative(~,aL,y)
            dCda=(aL-y);
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%
% Auxiliary functions
%%%%%%%%%%%%%%%%%%%%%
function y=sigma(x,activation)
if nargin<2
    activation='sigmoid';
end
switch activation
    case 'sigmoid'
        y=1./(1+exp(-x));
    case 'linear'
        y=x;
end
end

function y=sigma_prime(x,activation)
if nargin<2
    activation='sigmoid';
end
switch activation
    case 'sigmoid'
        sigmax=sigma(x);
        y=sigmax.*(1-sigmax);
    case 'linear'
        y=x.^0;
end
end




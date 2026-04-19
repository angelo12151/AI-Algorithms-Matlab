classdef perceptron < handle
    % percettrone di Rosenblatt
    % 
    % POPRIETA':
    %   weights
    %   bias
    % METODI
    % train(X,y)   
    %      Input
    %        X: dataset organizzato per righe
    %        y: vettore delle etichette {-1,1}
    % predict(X)
    %      Input
    %        X: punti (organizzati per righe)
    %           in cui valutare il percettrone
    %      Output
    %        y: vettore in {-1, 1} che classifica
    %           i punti in X
    properties
        weights
        bias
    end
    methods
        function obj=perceptron(n)
            obj.weights=zeros(n,1);
            obj.bias=0;
        end
        function it=train(obj,X,y)
            [m,~]=size(X);
            w=[obj.bias;obj.weights]; % vettore dei pesi (esteso)
            itmax=100*m;
            arresto=0;
            it=0; % numero di epoche
            while ~arresto && it<itmax
                it=it+1;
                arresto=1;
                for k=1:m
                    xk=[1 X(k,:)]';
                    if y(k)*(w'*xk)<=0
                        w=w+y(k)*xk;
                        arresto=0;
                    end
                end
            end
            if ~arresto
                warning('mancata convergenza')
            end
            obj.bias=w(1);
            obj.weights=w(2:end);
        end
        function y=predict(obj,X)
            y=sign(X*obj.weights+obj.bias);
        end
    end
end
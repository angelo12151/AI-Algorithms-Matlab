function varargout=percettrone(varargin)
% percettrone di Rosenblatt
% TRAINING
%   [w,b,it]=percettrone(X,y) 
% Input
%    X: dataset organizzato per righe 
%    y: vettore delle etichette {-1,1}
% Output
%    w: vettore dei pesi
%    b: bias
% PREDICTION
%   y=percettrone(X,w,b) 
% Input
%    X: punti (organizzati per righe) 
%       in cui valutare il percettrone 
%    w: vettore dei pesi
%    b: bias
% Output
%    y: vettore in {-1, 1} che classifica 
%       i punti in X

switch nargin 
    case 2
        % vogliamo addestrare il percettrone
        action='addestramento';
        X=varargin{1};
        y=varargin{2};
    case 3
        % vogliamo valutare il percettrone
        action='predizione';
        X=varargin{1};
        w=varargin{2};
        b=varargin{3};
    otherwise
        error('Input non consistente')
end
[m,n]=size(X);

switch action
    case 'predizione'
        y=sign(X*w+b);
        varargout{1}=y;
    case 'addestramento'
        w=zeros(n+1,1); % vettore dei pesi (esteso)
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
        varargout{1}=w(2:end);
        varargout{2}=w(1);
        varargout{3}=it;
end
clear
clf

%% ESEMPIO 1
% Mostra come una rete MLP riesce a separare due cluster che non siano
% linearmente separabili

%% set up training set
x1 = [0.1,0.2,0.3,0.4,0.1,0.6,0.4,0.9,0.6,0.5,0.9,0.4,0.7,0.9,0.3,0.5];
x2 = [0.1,0.8,0.4,0.9,0.5,0.9,0.2,0.8,0.3,0.8,0.2,0.4,0.6,0.5,0.7,0.1];
y = [ones(1,8) zeros(1,8); zeros(1,8) ones(1,8)];
x=[x1;x2];
N=length(x1);
training_data=cell(2,N);
for k=1:N
    training_data{1,k}=x(:,k);
    training_data{2,k}=y(:,k);
end
%% configure and train the neural network
rng(0) % per la riproducibilita'
net=anetwork([2 2 3 2]);
epochs=1.5e3;
mini_batch_size=3;
eta=0.6;
tic
% no graphical output
%net.sgd(training_data,epochs,mini_batch_size,eta)

% graphical output
net.sgd(training_data,epochs,mini_batch_size,eta,training_data)
fprintf('network trained in %g seconds \n',toc)
%% evaluate the net over the unit square
N=500;
Dx=1/N;
Dy=1/N;
xvals=[0:Dx:1];
yvals=[0:Dy:1];
for k1=1:N+1
    xk=xvals(k1);
    for k2=1:N+1
        yk=yvals(k2);
        xy=[xk;yk];
        ypred=net.feedforward(xy);
        Aval(k2,k1)=ypred(1);
        Bval(k2,k1)=ypred(2);
    end
end
[X,Y]=meshgrid(xvals,yvals);

%% plot results
figure(1)
clf
a2=subplot(1,1,1);
Mval=Aval>Bval;
contourf(X,Y,Mval,[0.5 0.5])
hold on
colormap([1 1 1;0.8,0.8,0.8])
plot(x1(1:8),x2(1:8),'or','MarkerSize',12,'LineWidth',4)
plot(x1(9:16),x2(9:16),'xb','MarkerSize',12,'LineWidth',4)
a2.XTick=[0 1];
a2.YTick=[0 1];
a2.FontWeight='Bold';
a2.FontSize=16;
xlim([0,1])
ylim([0,1])
hold off












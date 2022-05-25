clear all; clc;

subplot(3,3,1);
A = csvread('BTC.csv');
x = A(:,1); %eleminate last one day
plot(x,'m','LineWidth',1)
title('Original Data')
xlabel({'(a)'})
ylabel({'Bitcoin'})
xline(1094-28,'--r');
box off;
%set(gca,'FontSize',10)
set(gca,'FontWeight','bold','FontSize',12)
hold off

subplot(3,3,2);
A1 = csvread('BTC.csv');
A2 = csvread('predictedTest_stockBTC_mimicKalmanModified.csv');
A3 = csvread('EKF_BTC.csv');
A4 = csvread('predictedTest_BTC_AR.csv');
A5 = csvread('predictedTest_BTC_arima.csv');
% x = linspace(1,30,30);
% x1 = A1(1259-29:1259,1);
x = linspace(1,29,29);
x1 = A1(1094-28:1094,1); %eleminate last one day
x2 = A2(1:29,1);%LSTM : eleminate last one day
x3 = A3(1:29,1); %EKF : eliminate last one day
x4 = A4(2:30,1); %AR : eliminate first 1day
x5 = A5(2:30,1);
plot(x,x1,'m', x,x2,'b', x,x3,'r',x,x4,'g',x,x5,'c','LineWidth',1)
legend('True','LSTM','Kalman','AR','ARIMA')
legend boxoff   
title('True vs Predicted')
xlabel('(b)')
box off;
set(gca,'FontWeight','bold','FontSize',12)
hold off

subplot(3,3,3);
A1 = csvread('BTC.csv');
A2 = csvread('predictedTest_stockBTC_mimicKalmanModified.csv');
A3 = csvread('EKF_BTC.csv');
A4 = csvread('predictedTest_BTC_AR.csv');
A5 = csvread('predictedTest_BTC_arima.csv');
x = linspace(1,29,29);
%x1 = A1(1259-29:1258,1); %eleminate last one day
x1 = A1(1094-28:1094,1); %eleminate last one day
x2 = A2(1:29,1);%LSTM : eleminate last one day
x3 = A3(1:29,1); %EKF : eliminate last one day
x4 = A4(2:30,1); %AR : eliminate first 1day
x5 = A5(2:30,1);
plot(abs(x1-x2),'b','LineWidth',1); %LSTM difference
hold on
plot(abs(x1-x3),'r','LineWidth',1); 
hold on
plot(abs(x1-x4),'g','LineWidth',1); 
hold on
plot(abs(x1-x5),'c','LineWidth',1); 
legend('|True-LSTM|','|True-EKF|','|True-AR|','|True-ARIMA|')
legend boxoff   
title('Absolute Difference')
xlabel({'(c)'})
xlim([0,30]);
ylim([0,3500]);
box off;
set(gca,'FontWeight','bold','FontSize',12)
hold off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subplot(3,3,4);
A = csvread('ETH.csv');
x = A(:,1); %eleminate last one day
plot(x,'m','LineWidth',1)
xlabel({'(d)'})
ylabel({'Price($)','','Ethereum'})
xline(1094-28,'--r');
box off;
set(gca,'FontWeight','bold','FontSize',12)
hold off


% A1 = csvread('trueMSFT_5yr.csv');
% A2 = csvread('predictedTest_stockMS_mimicKalman_1228input1label.csv');
% A3 = csvread('kalman_stockMS.csv');
% x = linspace(1,30,30);
% x1 = A1(1259-29:1259,1);
% x2 = A2(:,1);
% x3 = A3(:,1);
% plot(x,x1,'m', x,x2,'b', x,x3,'r')
% xlabel('(e)')
% box off;
% set(gca,'FontWeight','bold','FontSize',10)
% hold off
subplot(3,3,5);
A1 = csvread('ETH.csv');
A2 = csvread('predictedTest_stockETH_mimicKalmanModified.csv');
A3 = csvread('EKF_ETH.csv');
A4 = csvread('predictedTest_ETH_AR.csv');
A5 = csvread('predictedTest_ETH_arima.csv');
x = linspace(1,29,29);
%x1 = A1(1259-29:1258,1); %eleminate last one day
x1 = A1(1094-28:1094,1); %eleminate last one day
x2 = A2(1:29,1);%LSTM : eleminate last one day
x3 = A3(1:29,1); %EKF : eliminate last one day
x4 = A4(2:30,1); %AR : eliminate first 1day
x5 = A5(2:30,1);
plot(x,x1,'m', x,x2,'b', x,x3,'r',x,x4,'g',x,x5,'c','LineWidth',1)
%legend('True','LSTM','Kalman','AR','ARIMA')
xlabel('(e)')
box off;
set(gca,'FontWeight','bold','FontSize',12)
hold off


% subplot(3,3,6);
% A1 = csvread('trueMSFT_5yr.csv');
% A2 = csvread('predictedTest_stockMS_mimicKalman_1228input1label.csv');
% A3 = csvread('kalman_stockMS.csv');
% x = linspace(1,30,30);
% x1 = A1(1259-29:1259,1);
% x2 = A2(:,1);
% x3 = A3(:,1);
% plot(abs(x1-x2),'b'); %LSTM difference
% hold on
% plot(-abs(x1-x3),'r'); 
% %legend('|True-LSTM|','-|True-Kalman|')
% xlabel({'(f)'})
% xlim([0,30]);
% ylim([-7,7]);
% box off;
% set(gca,'FontWeight','bold','FontSize',10)
% hold off
subplot(3,3,6);
A1 = csvread('ETH.csv');
A2 = csvread('predictedTest_stockETH_mimicKalmanModified.csv');
A3 = csvread('EKF_ETH.csv');
A4 = csvread('predictedTest_ETH_AR.csv');
A5 = csvread('predictedTest_ETH_arima.csv');
x = linspace(1,29,29);
%x1 = A1(1259-29:1258,1); %eleminate last one day
x1 = A1(1094-28:1094,1); %eleminate last one day
x2 = A2(1:29,1);%LSTM : eleminate last one day
x3 = A3(1:29,1); %EKF : eliminate last one day
x4 = A4(2:30,1); %AR : eliminate first 1day
x5 = A5(2:30,1);
plot(abs(x1-x2),'b','LineWidth',1); %LSTM difference
hold on
plot(abs(x1-x3),'r','LineWidth',1); 
hold on
plot(abs(x1-x4),'g','LineWidth',1); 
hold on
plot(abs(x1-x5),'c','LineWidth',1); 
%legend('|True-LSTM|','|True-EKF|','|True-AR|','|True-ARIMA|')
xlabel({'(f)'})
xlim([0,30]);
ylim([0,340]);
box off;
set(gca,'FontWeight','bold','FontSize',12)
hold off


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subplot(3,3,7);
A = csvread('ADA.csv');
x = A(:,1); %eleminate last one day
plot(x,'m','LineWidth',1)
xline(1094-28,'--r');
box off;
xlabel({'(g)'})
ylabel({'Cardano'})
set(gca,'FontWeight','bold','FontSize',12)
hold off


% A1 = csvread('trueGOOGL_5yr.csv');
% A2 = csvread('predictedTest_stockgoogle_mimicKalman_1228input1label.csv');
% A3 = csvread('kalman_stockgoogle.csv');
% x = linspace(1,30,30);
% x1 = A1(1259-29:1259,1);
% x2 = A2(:,1);
% x3 = A3(:,1);
% plot(x,x1,'m', x,x2,'b', x,x3,'r')
% xlabel({'(h)','','TIME(DAY)'})
% box off;
% set(gca,'FontWeight','bold','FontSize',10)
% hold off
subplot(3,3,8);
A1 = csvread('ADA.csv');
A2 = csvread('predictedTest_stockADA_mimicKalmanModified.csv');
A3 = csvread('EKF_ADA.csv');
A4 = csvread('predictedTest_ADA_AR.csv');
A5 = csvread('predictedTest_ADA_arima.csv');
x = linspace(1,29,29);
%x1 = A1(1259-29:1258,1); %eleminate last one day
x1 = A1(1094-28:1094,1); %eleminate last one day
x2 = A2(1:29,1);%LSTM : eleminate last one day
x3 = A3(1:29,1); %EKF : eliminate last one day
x4 = A4(2:30,1); %AR : eliminate first 1day
x5 = A5(2:30,1);
plot(x,x1,'m', x,x2,'b', x,x3,'r',x,x4,'g',x,x5,'c','LineWidth',1)
xlabel({'(h)','','Time(Day)'})
box off;
set(gca,'FontWeight','bold','FontSize',12)
hold off

% subplot(3,3,9);
% A1 = csvread('trueGOOGL_5yr.csv');
% A2 = csvread('predictedTest_stockgoogle_mimicKalman_1228input1label.csv');
% A3 = csvread('kalman_stockgoogle.csv');
% x = linspace(1,30,30);
% x1 = A1(1259-29:1259,1);
% x2 = A2(:,1);
% x3 = A3(:,1);
% plot(abs(x1-x2),'b'); %LSTM difference
% hold on
% plot(-abs(x1-x3),'r'); 
% %legend('|True-LSTM|','-|True-Kalman|')
% xlabel({'(i)'})
% xlim([0,30]);
% ylim([-135,135]);
% box off;
% set(gca,'FontWeight','bold','FontSize',10)
% hold off
subplot(3,3,9);
A1 = csvread('ADA.csv');
A2 = csvread('predictedTest_stockADA_mimicKalmanModified.csv');
A3 = csvread('EKF_ADA.csv');
A4 = csvread('predictedTest_ADA_AR.csv');
A5 = csvread('predictedTest_ADA_arima.csv');
x = linspace(1,29,29);
%x1 = A1(1259-29:1258,1); %eleminate last one day
x1 = A1(1094-28:1094,1); %eleminate last one day
x2 = A2(1:29,1);%LSTM : eleminate last one day
x3 = A3(1:29,1); %EKF : eliminate last one day
x4 = A4(2:30,1); %AR : eliminate first 1day
x5 = A5(2:30,1);
plot(abs(x1-x2),'b','LineWidth',1); %LSTM difference
hold on
plot(abs(x1-x3),'r','LineWidth',1); 
hold on
plot(abs(x1-x4),'g','LineWidth',1); 
hold on
plot(abs(x1-x5),'c','LineWidth',1); 
%legend('|True-LSTM|','|True-EKF|','|True-AR|','|True-ARIMA|')
xlabel({'(i)'})
xlim([0,30]);
ylim([0,0.36]);
box off;
set(gca,'FontWeight','bold','FontSize',12)
hold off
clear all; clc;

subplot(4,3,1);
epoch = [1,3,5,7,9,11,15,20,25,30,35,40,45,50];
x1 = [98.4,12.5,5.5,6.6,5.6,3.4,3.3,1.5,1.3,1.1,0.7,0.5,0.4,0.3];%LSTM
x2 = [1,1,1,1,1,1,1,1,1,1, 1, 1, 1, 1];%EKF
x3 = [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5, 0.5, 0.5, 0.5, 0.5];%AR
x4 = [0.15,0.15,0.15,0.15,0.15,0.15,0.15,0.15,0.15,0.15, 0.15, 0.15, 0.15, 0.15];%ARIMA
%x = A(:,1);
%plot(epoch,x,'g', epoch,x1, 'm')
plot(epoch,x1,'b', epoch,x2,'r', epoch,x3, 'g', epoch,x4,'c', 'LineWidth',1)
title('Apple')
xlabel({'(a)'})
ylabel({'RMSE'})
%xline(8449-199,'--r');
box off;
%set(gca,'FontSize',10)
set(gca,'FontWeight','bold','FontSize',12)
hold off

subplot(4,3,2);
epoch = [1,3,5,7,9,11,15,20,25,30,35,40,45,50];
x1 = [25746,5129,2324,2087,1818,1261,1330,736,634,383,895,240,356,265];%LSTM
x2 = [590,590,590,590,590,590,590,590,590,590, 590, 590, 590, 590];%EKF
x3 = [247,247,247,247,247,247,247,247,247,247, 247, 247, 247, 247];%AR
x4 = [219,219,219,219,219,219,219,219,219,219, 219, 219, 219, 219];%ARIMA
%x = A(:,1);
%plot(epoch,x,'g', epoch,x1, 'm')
plot(epoch,x1,'b', epoch,x2,'r', epoch,x3, 'g', epoch,x4,'c', 'LineWidth',1)
legend('LSTM','EKF','AR','ARIMA')
legend boxoff   
title('Bitcoin')
xlabel({'(b)'})
box off;
set(gca,'FontWeight','bold','FontSize',12)
hold off

subplot(4,3,3);
%A = csvread('trueoil.csv');
epoch = [1,3,5,7,9,11,15,20,25,30,35,40,45,50];
x1 = [493,142,101,92,42,39,37,18.7,16,14,6.5,5.8,18.5,3.7];%LSTM
x2 = [20.7,20.7,20.7,20.7,20.7,20.7,20.7,20.7,20.7,20.7, 20.7, 20.7, 20.7, 20.7];%EKF
x3 = [20.5,20.5,20.5,20.5,20.5,20.5,20.5,20.5,20.5,20.5, 20.5, 20.5, 20.5, 20.5];%AR
x4 = [20.78,20.78,20.78,20.78,20.78,20.78,20.78,20.78,20.78,20.78, 20.78, 20.78, 20.78, 20.78];;%ARIMA
%x = A(:,1);
%plot(epoch,x,'g', epoch,x1, 'm')
plot(epoch,x1,'b', epoch,x2,'r', epoch,x3, 'g', epoch,x4,'c', 'LineWidth',1)

title('Gold')
xlabel({'(c)'})
box off;
set(gca,'FontWeight','bold','FontSize',12)
hold off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subplot(4,3,4);
A1 = csvread('trueAAPL_5yr.csv');
A2 = csvread('predictedTest_stockapple_mimicKalmanModified_10epoch.csv');
A3 = csvread('EKF_apple.csv');
A4 = csvread('predictedTest_apple_AR.csv');
A5 = csvread('predictedTest_apple_ARIMA.csv');
x = linspace(1,29,29);
%x1 = A1(1258-29:1258,1);
x1 = A1(1230:1258,1);
x2 = A2(1:29,1);
x3 = A3(1:29,1);
x4 = A4(2:30,1);
x5 = A5(2:30,1);
plot(x,x1,'m', x,x2,'b', x,x3, 'r', x,x4,'g', x,x5,'c','LineWidth',1)
xlabel('(d)')
ylabel({'10 epoch'})
box off;
set(gca,'FontWeight','bold','FontSize',12)
hold off

subplot(4,3,5);
A1 = csvread('BTC.csv');
A2 = csvread('predictedTest_stockBTC_mimicKalmanModified_10epoch.csv');
A3 = csvread('EKF_BTC.csv');
A4 = csvread('predictedTest_BTC_AR.csv');
A5 = csvread('predictedTest_BTC_arima.csv');
x = linspace(1,29,29);
x1 = A1(1066:1094,1);
x2 = A2(1:29,1);
x3 = A3(1:29,1);
x4 = A4(2:30,1);
x5 = A5(2:30,1);
plot(x,x1,'m', x,x2,'b', x,x3, 'r', x,x4,'g', x,x5,'c','LineWidth',1)
%legend('True','LSTM','kalman')
xlabel('(e)')
box off;
set(gca,'FontWeight','bold','FontSize',12)
hold off

subplot(4,3,6);
A1 = csvread('truegold.csv');
A2 = csvread('predictedTest_cryptogold_mimicKalmanModified_10epoch.csv');
A3 = csvread('EKF_gold.csv');
A4 = csvread('predictedTest_gold_AR.csv');
A5 = csvread('predictedTest_gold_arima.csv');
x = linspace(1,29,29);
x1 = A1(818:846,1);
x2 = A2(1:29,1);
x3 = A3(1:29,1);
x4 = A4(2:30,1);
x5 = A5(2:30,1);
plot(x,x1,'m', x,x2,'b', x,x3, 'r', x,x4,'g', x,x5,'c','LineWidth',1)
%legend('True','LSTM','kalman')
xlabel('(f)')
box off;
set(gca,'FontWeight','bold','FontSize',12)
hold off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subplot(4,3,7);
A1 = csvread('trueAAPL_5yr.csv');
A2 = csvread('predictedTest_stockapple_mimicKalmanModified_20epoch.csv');
A3 = csvread('EKF_apple.csv');
A4 = csvread('predictedTest_apple_AR.csv');
A5 = csvread('predictedTest_apple_ARIMA.csv');
x = linspace(1,29,29);
x1 = A1(1230:1258,1);
x2 = A2(1:29,1);
x3 = A3(1:29,1);
x4 = A4(2:30,1);
x5 = A5(2:30,1);
plot(x,x1,'m', x,x2,'b', x,x3, 'r', x,x4,'g', x,x5,'c','LineWidth',1)
%legend('True','LSTM')
xlabel('(g)')
ylabel({'Price','','20 epoch'})
box off;
set(gca,'FontWeight','bold','FontSize',12)
hold off

subplot(4,3,8);
A1 = csvread('BTC.csv');
A2 = csvread('predictedTest_stockBTC_mimicKalmanModified_20epoch.csv');
A3 = csvread('EKF_BTC.csv');
A4 = csvread('predictedTest_BTC_AR.csv');
A5 = csvread('predictedTest_BTC_arima.csv');
x = linspace(1,29,29);
x1 = A1(1066:1094,1);
x2 = A2(1:29,1);
x3 = A3(1:29,1);
x4 = A4(2:30,1);
x5 = A5(2:30,1);
plot(x,x1,'m', x,x2,'b', x,x3, 'r', x,x4,'g', x,x5,'c','LineWidth',1)
legend('True','LSTM','EKF','AR','ARIMA')
legend boxoff   
%legend('True','LSTM')
xlabel('(h)')
box off;
set(gca,'FontWeight','bold','FontSize',12)
hold off

subplot(4,3,9);
A1 = csvread('truegold.csv');
A2 = csvread('predictedTest_cryptogold_mimicKalmanModified_10epoch.csv');
A3 = csvread('EKF_gold.csv');
A4 = csvread('predictedTest_gold_AR.csv');
A5 = csvread('predictedTest_gold_arima.csv');
x = linspace(1,29,29);
x1 = A1(818:846,1);
x2 = A2(1:29,1);
x3 = A3(1:29,1);
x4 = A4(2:30,1);
x5 = A5(2:30,1);
plot(x,x1,'m', x,x2,'b', x,x3, 'r', x,x4,'g', x,x5,'c','LineWidth',1)
%legend('True','LSTM')
xlabel('(i)')
box off;
set(gca,'FontWeight','bold','FontSize',12)
hold off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subplot(4,3,10);
A1 = csvread('trueAAPL_5yr.csv');
A2 = csvread('predictedTest_stockapple_mimicKalmanModified_50epoch.csv');
A3 = csvread('EKF_apple.csv');
A4 = csvread('predictedTest_apple_AR.csv');
A5 = csvread('predictedTest_apple_ARIMA.csv');
x = linspace(1,29,29);
x1 = A1(1230:1258,1);
x2 = A2(1:29,1);
x3 = A3(1:29,1);
x4 = A4(2:30,1);
x5 = A5(2:30,1);
plot(x,x1,'m', x,x2,'b', x,x3, 'r', x,x4,'g', x,x5,'c','LineWidth',1)
xlabel('(j)')
ylabel({'50 epoch'})
box off;
set(gca,'FontWeight','bold','FontSize',12)
hold off

subplot(4,3,11);
A1 = csvread('BTC.csv');
A2 = csvread('predictedTest_stockBTC_mimicKalmanModified_50epoch.csv');
A3 = csvread('EKF_BTC.csv');
A4 = csvread('predictedTest_BTC_AR.csv');
A5 = csvread('predictedTest_BTC_arima.csv');
x = linspace(1,29,29);
x1 = A1(1066:1094,1);
x2 = A2(1:29,1);
x3 = A3(1:29,1);
x4 = A4(2:30,1);
x5 = A5(2:30,1);
plot(x,x1,'m', x,x2,'b', x,x3, 'r', x,x4,'g', x,x5,'c','LineWidth',1)
%legend('True','LSTM')
xlabel({'(k)','','Time(Day)'})
box off;
set(gca,'FontWeight','bold','FontSize',12)
hold off

subplot(4,3,12);
A1 = csvread('truegold.csv');
A2 = csvread('predictedTest_cryptogold_mimicKalmanModified_10epoch.csv');
A3 = csvread('EKF_gold.csv');
A4 = csvread('predictedTest_gold_AR.csv');
A5 = csvread('predictedTest_gold_arima.csv');
x = linspace(1,29,29);
x1 = A1(818:846,1);
x2 = A2(1:29,1);
x3 = A3(1:29,1);
x4 = A4(2:30,1);
x5 = A5(2:30,1);
plot(x,x1,'m', x,x2,'b', x,x3, 'r', x,x4,'g', x,x5,'c','LineWidth',1)
%legend('True','LSTM')
xlabel('(l)')
box off;
set(gca,'FontWeight','bold','FontSize',12)
hold off
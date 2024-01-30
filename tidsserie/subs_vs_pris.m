clear all 
close all
clc

addpath functions
addpath data

data = readtable('vnv.csv');

% Convert the first column to datetime format
date_format = 'dd-mmm-yyyy';
data.date = datetime(data.(1), 'InputFormat', date_format);
data.price = data.(2);
data.subs1 = data.(3);
data.subs2 = data.(4);
%% First look, price vs substansvärde
plot(data.date, data.price)
hold on
plot(data.date, data.subs1)
legend('price','substansvärde')
%% Maybe Ar(1)
analyzeACF(data.price, 50)
%%
model_data = iddata(data.price);
model_init = idpoly([1 1], [], []);

model_armax = pem(model_data, model_init)

e_hat = resid(model_armax, model_data).y(3:end);

checkIfWhite(e_hat)
analyzeACF(e_hat, 50, 0.05, "process");


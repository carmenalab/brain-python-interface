close all; clear; clc;

% from .ns5 --> .hdf --> .mat
load('/Users/sdangi/code/ibmi/blackrock/data/cont.ns5.mat')

% from nPlay --> cbpy --> .mat
load('/Users/sdangi/code/bmi3d/riglib/blackrock/cbpy_lfp_data.mat')

channels = [5, 6];
chan_ind = 1;
chan = channels(chan_ind);

ns5_data = eval('chan00005');
cbpy_data = data(chan_ind,1:15000);

seg = cbpy_data;
seg_len = length(seg);
channel_data = double(ns5_data(1:30:end));

figure();
subplot(2,1,1)
plot(seg);
subplot(2,1,2)
plot(channel_data)

figure();
n_subplots = 2; subplot_num = 1;

y = conv(fliplr(seg), channel_data);
subplot(n_subplots,1,subplot_num); subplot_num = subplot_num + 1;
plot(y);
title('sliding inner product');

[max_val, ind] = max(y);

c = 1;
subplot(n_subplots,1,subplot_num); subplot_num = subplot_num + 1;
hold on;
plot(channel_data(ind-seg_len+1:ind),'g')
plot(seg);
hold off;
title('best match (different gains)');
legend('ns5', 'cbpy');



% % gains = 400:0.05:420;
% gains = 0.5:0.001:1.5; %10:0.1:500;
% mse = zeros(length(gains), 1);
% 
% for i = 1:length(gains)
%     gain = gains(i);
%     mse(i) = sum((gain*channel_data(ind-seg_len+1:ind)' - seg).^2);
% end
% 
% [min_mse, min_ind] = min(mse);
% 
% true_gain = gains(min_ind);
% disp(['true gain: ', num2str(true_gain)]);
% 
% subplot(n_subplots,1,subplot_num); subplot_num = subplot_num + 1;
% plot(gains, mse);
% title('mse curve to find gain');
% 
% figure(13);
% plot(c*channel_data(ind-seg_len+1:ind) - seg','g')

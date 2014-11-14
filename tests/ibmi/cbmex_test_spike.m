close all; clear; clc;

addpath('/usr/local/CereLink');

% open library
cbmex('open','inst-addr','192.168.137.128','inst-port',51001,'central-addr','255.255.255.255','central-port',51002);
 
[active_state, config_vector_out] = cbmex('trialconfig', 1);

n_chan = 128;

n_secs = 10;
loop_time = 0.1;
n_itrs = round(n_secs/loop_time);

% ts = cell(n_chan,n_itrs);
ts = cell(n_chan, 1);
for chan = 1:n_chan
    ts{chan} = [];
end

for itr = 1:n_itrs
    t_start = GetSecs(); 
    [spike_data, t_most_recent_clear, continuous_data] = cbmex('trialdata',1); % read some data
%     [time, continuous_data] = cbmex('trialdata',1); % read some data
%     disp(continuous_data);
    
%     for chan = 1:n_chan
%         ts{chan, itr} = spike_data{chan,2};
%     end
    for chan = 1:n_chan
        ts{chan} = [ts{chan}; spike_data{chan,2}];
    end

    t_elapsed = GetSecs() - t_start;
    fprintf('t_elapsed: %3.3f secs', t_elapsed);
    if t_elapsed < loop_time
        WaitSecs(loop_time - t_elapsed);
    end
    fprintf(', final loop time: %3.3f secs\n', GetSecs()-t_start)
end

% cbmex('close');

close all;
clear variables;
 
f_disp = 0:0.1:15;       % the range of frequency to show spectrum over. 
% Use f_disp = [] if you want the entire spectrum
 
collect_time = 0.5; % collect samples for this time
display_period = 1; % display spectrum every this amount of time
 
% cbmex('open'); % open library
cbmex('open','inst-addr','192.168.137.128','inst-port',51001,'central-addr','255.255.255.255','central-port',51002);

proc_fig = figure; % main display
set(proc_fig, 'Name', 'Close this figure to stop');
xlabel('frequency (Hz)');
ylabel('magnitude (dB)');
 
% cbmex('trialconfig', 1); % empty the buffer
 
begchan = 5;
begmask = 0;
begval  = 0;
endchan = 8;
endmask = 0;
endval  = 0;

config_vector_in = [begchan begmask begval endchan endmask endval];
[active_state, config_vector_out] = cbmex('trialconfig', 1, config_vector_in);

t_disp0 = tic; % display time
t_col0  = tic; % collection time
bCollect = true; % do we need to collect
 % while the figure is open
while (ishandle(proc_fig)) 
    
    if (bCollect)
        et_col = toc(t_col0); % elapsed time of collection
        if (et_col >= collect_time)

            [spike_data, t_buf1, continuous_data] = cbmex('trialdata',1); % read some data
            % [time, continuous_data] = cbmex('trialdata',1); % read some data

            nGraphs = size(continuous_data,1); % number of graphs
            % if the figure is still open
            if (ishandle(proc_fig))
                % graph all 
                for ii=1:nGraphs
                    fs0 = continuous_data{ii,2};
                    % get the ii'th channel data
                    data = continuous_data{ii,3};
                    % number of samples to run through fft
                    collect_size = min(size(data), collect_time * fs0);
                    x = data(1:collect_size);
                    %uncomment to see the full rang
                    if isempty(f_disp)
                        [psd, f] = periodogram(double(x),[],'onesided',512,fs0);
                    else
                        [psd, f] = periodogram(double(x),[],f_disp,fs0);
                    end
                    subplot(nGraphs,1,ii,'Parent',proc_fig);
                    plot(f, 10*log10(psd), 'b');title(sprintf('fs = %d t = %f', fs0, t_buf1));
                    xlabel('frequency (Hz)');ylabel('magnitude (dB)');
                end
                drawnow;
            end
            bCollect = false;
        end
    end
    
    et_disp = toc(t_disp0); % elapsed time since last display
    if (et_disp >= display_period)
        t_col0  = tic; % collection time
        t_disp0 = tic; % restart the period
        bCollect = true; % start collection
    end
end
cbmex('close'); % always close

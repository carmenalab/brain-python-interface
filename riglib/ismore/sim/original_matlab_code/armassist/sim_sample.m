% Sample program for simulation of ArmAssist without\with external force

close all
clear
clc
tstep = 0.01;  % sampling frequeny of control loop - you can change it
k_ref = 10; % for reference updating every 0.1 sec
ftime = 10;    % final time of simulation - you can change it
n_substeps = 50;

% mechanical constants of robot - don´t change values
m = 3.05;     % m - Robot mass (kg)
Iz = 0.067;   % Iz - Robot moment inertia (kgm2) ! this is not actual value
R = 0.0254;   % R - Wheel radius (m)
L1 = 0.130;   % L1 - Distance from the center mouse to front wheel.
L2 = 0.153;   % L2 - Distance from the center mouse to rear right (left) wheel.
n = 19;       % n - Gear ratio
J0 = 0.003;   % combined inertia of the motor, gear train and wheel referred to the motor shaft ! this is not acutal value
b0 = 0;       % viscous-friction coefficient of the motor, gear and wheel combination

% matrices used in the model - don´t change them. If you need to change, please let me know.
H = [1/m 0 0;0 1/m 0;0 0 1/Iz];
B = [-1 cos(pi/4) cos(pi/4);0 -sin(pi/4) sin(pi/4);L1 L2*cos(pi*12/180) L2*cos(pi*12/180)];
G = (eye(3)+H*B*B'*n^2*J0/R^2);

% bf - vector including 3 variables of dynamic model of ArmAssist in the body frame
% bf(1) --> u - velocity component in x direction in the body frame (m/s)
% bf(2) --> v - velocity component in y direction in the body frame (m/s)
% bf(3) --> r - anglular rate of body rototation (rad/s)
bf = [0 0 0]'; % initial vector of bf

% wf - vector including 3 variables of ArmAssist in the world frame
% wf(1) --> x - x position in the world frame (m)
% wf(2) --> y - y position in the world frame (m)
% wf(3) --> psi - robot orientation angle (rad)
wf = [0 0 0]'; % initial vector of wf

ie_bf = [0;0;0]; % initial values for Integral control

%start simulation

tic;
t_vec = 0:tstep:ftime;
for i = 1:length(t_vec)
    t = t_vec(i);
    
    %external force vector. Caution-external force vector should be defined in the body frame.
    ex_f_xr = 0;% external force in the x-direction at the body frame (XR in the figure in the description)
    ex_f_yr = 0;% external force in the y-direction at the body frame (YR in the figure in the description)
    ex_t_zr = 0;% Caution: external torque in the z-direction at the body frame (ZR in the figure in the description)
    ex_f = [ex_f_xr; ex_f_yr; ex_t_zr];
    
    %desired values in the global frame
    if k_ref == 10
        des_x_vel = 0.05;% m/s
        des_y_vel = 0.05*sin(t);% m/s
        des_z_ang_vel = 0; %rad/s
        k_ref = 0;
    end
    k_ref = k_ref + 1;
    
    %Transform desired vector in the global frame to that in the body frame
    K_M=[cos(wf(3)) sin(wf(3)) 0;-sin(wf(3)) cos(wf(3)) 0;0 0 1]*([des_x_vel;des_y_vel;des_z_ang_vel]);
    
    %design of input - you can design those values as you need.
    % PI control
    e_bf = [K_M(1);K_M(2);K_M(3)] - bf; % error between desired and actual
    ie_bf = ie_bf + e_bf*tstep;      % integration of error
    
    KP  = [-10. 0. 0.;0. -20. 0.;0. 0. 20.]; % P gain matrix
    TI =  [0.1 0. 0.;0. 0.1 0.;0. 0. 0.1];   % I gain matrix
    
    torq = KP*(e_bf+TI*ie_bf);   % PI Control  torq(1)-motor 1 torque, torq(2)-motor 2 torque, torq(3)-motor 3 torque
    
    %%% mobile robot dynamics %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for ii=1:n_substeps  % the for-structure is utilized in order to make the simulation continuous
        % integration in body frame
        [bf,bf_dot] = rungekutta_body_frame(torq,bf,G,H,B,R,b0,n,ex_f,tstep/n_substeps);
        
        %integration in world frame
        [wf,wf_dot] = rungekutta_world_frame(wf,bf,tstep/n_substeps);
    end
    disp('bf');
    disp(bf);
    
    %%%%% data save   %%%%%%%%%%%%%%%%%%%%%%%%%%
    u(i) = e_bf(1);   % velocity and angular rate components in the body frame
    v(i) = e_bf(2);
    r(i) = e_bf(3);
    
    x(i) = wf(1);    % position and orientation components in the world frame
    y(i) = wf(2);
    psi(i) = wf(3);
    
    des_x_v(i) = des_x_vel;   % shaped refernce velocity
    des_y_v(i) = des_y_vel;
    des_psi_v(i) = des_z_ang_vel;
    
    x_v(i) = wf_dot(1);    % velocity and angular velocity components in the world frame
    y_v(i) = wf_dot(2);
    psi_v(i) = wf_dot(3);
end
toc;

k = 0:tstep:ftime;

% plot of desired and actual translational and rotational velocities
figure(1)
subplot(311)
plot (k,des_x_v',k,x_v'),title ('x direction'),ylabel('m/s'),legend('desired velocity','real velocity',4)
subplot(312)
plot (k,des_y_v',k,y_v'),title ('y direction'),ylabel('m/s'),legend('desired velocity','real velocity',4)
subplot(313)
plot (k,des_psi_v',k,psi_v'),title ('rotation about z axis'),ylabel('rad/s'),xlabel('time(sec)'),legend('desired velocity','real velocity',4)

% plot of ArmAssist movement during the control
figure(2)
axis([-0.5 1 -0.5 0.5]);

i = 1;
cmc_x = (x(i)-L1*sin(psi(i)));
cmc_y = (y(i)+L1*cos(psi(i)));
wh1_x = (x(i)+L2*sin(32.5/180*pi+psi(i)));
wh1_y = (y(i)-L2*cos(32.5/180*pi+psi(i)));
wh2_x = (x(i)-L2*sin(32.5/180*pi-psi(i)));
wh2_y = (y(i)-L2*cos(32.5/180*pi-psi(i)));
wh3_x = x(i);
wh3_y = y(i);

hold on;
h_tri = plot([wh1_x, wh2_x, wh3_x, wh1_x], [wh1_y, wh2_y, wh3_y, wh1_y]);
hold off;

tic;
for i = 1:length(t_vec)
    t_start = GetSecs();
    
    cmc_x = (x(i)-L1*sin(psi(i)));
    cmc_y = (y(i)+L1*cos(psi(i)));
    wh1_x = (x(i)+L2*sin(32.5/180*pi+psi(i)));
    wh1_y = (y(i)-L2*cos(32.5/180*pi+psi(i)));
    wh2_x = (x(i)-L2*sin(32.5/180*pi-psi(i)));
    wh2_y = (y(i)-L2*cos(32.5/180*pi-psi(i)));
    wh3_x = x(i);
    wh3_y = y(i);
    
    set(h_tri,'xdata',[wh1_x, wh2_x, wh3_x, wh1_x],'ydata',[wh1_y, wh2_y, wh3_y, wh1_y]);
    
    %       set(time_h, 'String', sprintf('Time: %3.1f s', i*tstep));
    
    drawnow;
    
    t_elapsed = GetSecs()-t_start;
    WaitSecs(tstep - t_elapsed);
end
toc;

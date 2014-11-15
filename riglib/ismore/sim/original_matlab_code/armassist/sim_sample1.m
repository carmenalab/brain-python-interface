% Sample program for simulation of ArmAssist without\with external force

clear
tstep = 0.01;  % sampling frequeny of control loop - you can change it
ftime = 5;    % final time of simulation - you can change it

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

i = 1; % for data saving
ie_bf = [0;0;0]; % initial values for Integral control

%start simulation

for t=0:tstep:ftime
 
   %external force vector. Caution-external force vector should be defined in the body frame.
   ex_f_xr = 0;% external force in the x-direction at the body frame (XR in the figure in the description)
   ex_f_yr = 0;% external force in the y-direction at the body frame (YR in the figure in the description)
   ex_t_zr = 0;% Caution: external torque in the z-direction at the body frame (ZR in the figure in the description)
   ex_f = [ex_f_xr;ex_f_yr;ex_t_zr]; 
   
   %desired values in the global frame
   des_x_vel = 0.05;% m/s
   des_y_vel = 0.05*sin(t);% m/s
   des_z_ang_vel = 0; %rad/s 

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
   %intergration in body frame
   for ii=1:50  % the for-structure is utilized in order to make the simulation continuos
   [bf,bf_dot] = rungekutta_body_frame(torq,bf,G,H,B,R,b0,n,ex_f,tstep/50); 
   end
   
   %intergration in world frame
   for ii=1:50
   [wf,wf_dot] = rungekutta_world_frame(wf,bf,tstep/50); 
   end

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
   
   i = i + 1;
end

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
ik = 20; % for plotting every 20*sampling frequency
for iii=1:501 
    if ik == 20
 % plot of 3 wheel locations and location of the center mouse camera   
 plot((x(iii)-L1*sin(psi(iii))),(y(iii)+L1*cos(psi(iii))),'mo',(x(iii)+L2*sin(32.5/180*pi+psi(iii))),(y(iii)-L2*cos(32.5/180*pi+psi(iii))),'ro',(x(iii)-L2*sin(32.5/180*pi-psi(iii))),(y(iii)-L2*cos(32.5/180*pi-psi(iii))),'bo',x(iii),y(iii),'r*');

 % to draw the trinalge that is made by connecting 3 wheel points
   line([x(iii)-L1*sin(psi(iii)),x(iii)+L2*sin(32.5/180*pi+psi(iii)),x(iii)-L2*sin(32.5/180*pi-psi(iii)),x(iii)-L1*sin(psi(iii))],[y(iii)+L1*cos(psi(iii)),y(iii)-L2*cos(32.5/180*pi+psi(iii)),y(iii)-L2*cos(32.5/180*pi-psi(iii)),y(iii)+L1*cos(psi(iii))]);
 ik = 0;
end
ik = ik +1;
end





% Main program for simulation of pronosupination mechanism

tstep = 0.1;  % sampling frequeny - 
ftime = 5;    % final time of simulation - sec

% mechanical constants of pronosupination mechanism
Im = 43.3; %g.cm^2; // inertia of a motor shaft - fixed 
bm = 0;   % (g.cm^2/s^2)/(rad/s) - damping (viscous friction) coefficient of a motor -fixed
n1 = 5.2; % gear ration of gear box - fixed
IG = 1.25; % g.cm^2; inertia of a gear box -fixed
bG = 0;  % damping (viscous friction) coefficient of a gear box - fixed
n2 = 8;  % gear ration between a gear box and a pronosupination arc piece - fixed
Ips = 8064; % g.cm^2; inertia of a pronosupination arc piece - fixed
Ih = 58000; % g.cm^2; inertial of human hand + forearm - changeable depending on a subject
Ipst = Ips + Ih; % g.cm^2 ;total inertial of pronosupination arc piece + (human hand + forearm)

% effective inertia and damping

I_eff = (Im + IG/(n1)^2 + Ipst/(n1*n2)^2);
b_eff = (bm + bG/(n1)^2);

% state vector X composed of motor angle (theta) and angular velocity, x =[theta dtheta]'
x = [0 0]'; % initial value x(1) - theta (radian), x(2)-angular velocity (radian/sec)

i = 1; % for data saving

ie = 0; % first integral term

for t=0:tstep:ftime
   
   %desired angular velocity of pronosupination
   des_ang_vel = 0.1*sin(2*t); %radian/sec
   
   %PI control
    err_ang_vel = des_ang_vel*(n1*n2) - x(2); % need to transform the desired angular velocity of pronosupination into that of motor
    ie_ = ie + err_ang_vel*tstep;
    
   %PI gains
    Kp = 600; 
    Ki = 1300;
   
    mtorq = Kp*(err_ang_vel+Ki*ie);
    
   %
   %%%% pronosupination mechanism dynamics %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   %%intergration 
   for ii=1:50
   [x,x_dot] = rungekutta(x,mtorq,I_eff,b_eff,tstep/50); 
   end
   
   %%%%%%% data save   %%%%%%%%%%%%%%%%%%%%%%%%%%
   des_angular_velocity(i) =  des_ang_vel;    
   real_angular_velocity(i) = x(2)/(n1*n2); % motor angular velocity --> pronosupination angular velocity
   
   angle_ps(i) = x(1)/(n1*n2); % pronosupination angle.
   motor_torque(i) = mtorq;
   
   i=i+1;
   
end

k = 0:tstep:ftime;

%plot degree and degree/sec
subplot(311)
plot (k,des_angular_velocity'*180/pi,k,real_angular_velocity'*180/pi),title ('PS angular velocity'),ylabel('deg/s')
subplot(312)
plot (k,angle_ps'*180/pi),title ('PS rotation angle'),ylabel('deg')
subplot(313)
plot (k,motor_torque'),title ('motor torque'),ylabel('mNm'),xlabel('time(sec)')


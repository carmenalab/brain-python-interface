function [x_dot] = dyn_eq_pronosupination(x,mtorq,I_eff,b_eff)
    
% state vector x composed of motor angle (theta) and angular velocity, x =[theta dtheta]'
% mtorq - motor torque
% I_eff - effective inertia 
% b_eff - effective damping (viscous friction) coefficient

th = x(1);
dth = x(2);

ddth =(mtorq - b_eff*dth)/I_eff;  %dynamics of the pronosupination mechanism.

x_dot = [dth;ddth];

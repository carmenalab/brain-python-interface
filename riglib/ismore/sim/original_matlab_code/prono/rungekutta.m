function [x,x_dot]=rungekutta(x,mtorq,I_eff,b_eff,tstep)

% state vector q composed of motor angle (theta) and angular velocity, x =[theta dtheta]'
% mtorq - motor torque
% I_eff - effective inertia 
% b_eff - effective damping (viscous friction) coefficient

X = x;

x_dot1 = dyn_eq_pronosupination(X,mtorq,I_eff,b_eff);
x = X + x_dot1*tstep/2;
x_dot2 = dyn_eq_pronosupination(x,mtorq,I_eff,b_eff);
x = X + x_dot2*tstep/2;
x_dot3 = dyn_eq_pronosupination(x,mtorq,I_eff,b_eff);
x = X + x_dot3*tstep;
x_dot4 = dyn_eq_pronosupination(x,mtorq,I_eff,b_eff);
X = X+tstep*((x_dot1+x_dot4)/2+x_dot2+x_dot3)/3;

x = X;
x_dot = x_dot1;

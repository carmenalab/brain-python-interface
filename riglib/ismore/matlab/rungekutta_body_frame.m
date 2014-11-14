% rungekutta integration of dynamics in the body frame
function [q,q_dot]=rungekutta_body_frame(tor,bf,G,H,B,R,b0,n,f_ex,tstep)
X = bf;
torque = tor;

x_dot1 = dyn_eq_body_frame(torque,X,G,H,B,R,b0,n,f_ex);
x = X + x_dot1*tstep/2;
x_dot2 = dyn_eq_body_frame(torque,x,G,H,B,R,b0,n,f_ex);
x = X + x_dot2*tstep/2;
x_dot3 = dyn_eq_body_frame(torque,x,G,H,B,R,b0,n,f_ex);
x = X + x_dot3*tstep;
x_dot4 = dyn_eq_body_frame(torque,x,G,H,B,R,b0,n,f_ex);
X = X+tstep*((x_dot1+x_dot4)/2+x_dot2+x_dot3)/3;

q = X;
q_dot = x_dot1;

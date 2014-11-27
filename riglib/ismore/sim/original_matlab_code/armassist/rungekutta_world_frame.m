% rungekutta integration of kinematics in the world frame
function [q,q_dot]=rungekutta_world_frame(wf,bf,tstep)
X = wf;

x_dot1 = kine_eq_robot(bf,X(3));
x = X + x_dot1*tstep/2;
x_dot2 = kine_eq_robot(bf,x(3));
x = X + x_dot2*tstep/2;
x_dot3 = kine_eq_robot(bf,x(3));
x = X + x_dot3*tstep;
x_dot4 = kine_eq_robot(bf,x(3));
X = X+tstep*((x_dot1+x_dot4)/2+x_dot2+x_dot3)/3;

q = X;
q_dot = x_dot1;

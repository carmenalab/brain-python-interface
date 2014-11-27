% dynamic model of ArmAssist in the body frame
function [bf_dot] = dyn_eq_body_frame(torq,bf,G,H,B,R,b0,n,f_ex)   

% volt - input volts vector of 3 motors: volt(i),i=1,2,3 (V)
% bf - vector including 3 variables of dynamic model of the mobile robot in the body frame
% bf(1) --> u - velocity component in x direction in the body frame (m/s) 
% bf(2) --> v - velocity component in y direction in the body frame (m/s)
% bf(3) --> r - anglular rate of body rototation (rad/s) 
% f_ex --> external force in the body frame

% bf_dot = inv(G)*[bf(3)*bf(2);-bf(3)*bf(1);0]-inv(G)*H*B*B'*b0*n^2/R^2*bf+inv(G)*H*B*n/R*torq+f_ex; % equation of motion
bf_dot = G\([bf(3)*bf(2);-bf(3)*bf(1);0]-H*B*B'*b0*n^2/R^2*bf+H*B*n/R*torq)+f_ex; % equation of motion



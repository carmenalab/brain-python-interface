 % kinematics of ArmAssist in world frame
function [wf_dot] = kine_eq_robot(bf,a)  

% bf - vector including 3 variables of dynamic model of the mobile robot in the body frame
% bf(1) --> u - velocity component in x direction in the body frame (m/s) 
% bf(2) --> v - velocity component in y direction in the body frame (m/s)
% bf(3) --> r - anglular rate of body rototation (rad/s) 
% a - robot orientation angle (rad) in world frame

wf_dot = [cos(a) -sin(a) 0;sin(a) cos(a) 0;0 0 1]*bf;

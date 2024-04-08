% Unconstrained minimization

% Meyer's (reformulated) problem
p0=[8.85, 4.0, 2.5];

x(1:4)=[34.780, 28.610, 23.650, 19.630];
x(5:8)=[16.370, 13.720, 11.540, 9.744];
x(9:12)=[8.261, 7.030, 6.005, 5.147];
x(13:16)=[4.427, 3.820, 3.307, 2.872];

options=[1E-03, 1E-15, 1E-15, 1E-20, 1E-06];
% arg1, arg2 demonstrate additional data passing to meyer/jacmeyer
arg1=[17];
arg2=[27];

%[ret, popt, info]=levmar('meyer', 'jacmeyer', p0, x, 200, options, arg1, arg2);

%[ret, popt, info, covar]=levmar('meyer', 'jacmeyer', p0, x, 200, options, arg1, arg2);
[ret, popt, info, covar]=levmar('meyer', p0, x, 200, options, 'unc', arg1, arg2);
disp('Meyer''s (reformulated) problem');
popt


% Linear constraints

% Boggs-Tolle problem 3
p0=[2.0, 2.0, 2.0, 2.0, 2.0];
x=[0.0, 0.0, 0.0, 0.0, 0.0];
options=[1E-03, 1E-15, 1E-15, 1E-20];
adata=[];

A=[1.0, 3.0, 0.0, 0.0, 0.0;
   0.0, 0.0, 1.0, 1.0, -2.0;
   0.0, 1.0, 0.0, 0.0, -1.0];
b=[0.0, 0.0, 0.0]';

[ret, popt, info, covar]=levmar('bt3', 'jacbt3', p0, x, 200, options, 'lec', A, b, adata);
disp('Boggs-Tolle problem 3');
popt


% Box constraints

% Hock-Schittkowski problem 01
p0=[-2.0, 1.0];
x=[0.0, 0.0];
lb=[-realmax, -1.5];
ub=[realmax, realmax];
options=[1E-03, 1E-15, 1E-15, 1E-20];

[ret, popt, info, covar]=levmar('hs01', 'jachs01', p0, x, 200, options, 'bc', lb, ub);
disp('Hock-Schittkowski problem 01');
popt


% Box and linear constraints

% Hock-Schittkowski modified problem 52
p0=[2.0, 2.0, 2.0, 2.0, 2.0];
x=[0.0, 0.0, 0.0, 0.0];
lb=[-0.09, 0.0, -realmax, -0.2, 0.0];
ub=[realmax, 0.3, 0.25, 0.3, 0.3];
A=[1.0, 3.0, 0.0, 0.0, 0.0;
   0.0, 0.0, 1.0, 1.0, -2.0;
   0.0, 1.0, 0.0, 0.0, -1.0];
b=[0.0, 0.0, 0.0]';
options=[1E-03, 1E-15, 1E-15, 1E-20];

[ret, popt, info, covar]=levmar('modhs52', 'jacmodhs52', p0, x, 200, options, 'blec', lb, ub, A, b);
disp('Hock-Schittkowski modified problem 52');
popt

% Hock-Schittkowski modified problem 235
p0=[-2.0, 3.0, 1.0];
x=[0.0, 0.0];
lb=[-realmax, 0.1, 0.7];
ub=[realmax, 2.9, realmax];
A=[1.0, 0.0, 1.0;
   0.0, 1.0, -4.0];
b=[-1.0, 0.0]';
options=[1E-03, 1E-15, 1E-15, 1E-20];

[ret, popt, info, covar]=levmar('mods235', p0, x, 200, options, 'blec', lb, ub, A, b);
disp('Hock-Schittkowski modified problem 235');
popt


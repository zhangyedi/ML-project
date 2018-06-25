% This is test script for logistic regression 

close all;
clear;
clc;

% nba_data is a matrix 
% each row denotes the performance of a certain NBA team, which contains 
% scores, assits, backboard, etc. The last column indicate the team win 
% champions or not
load nbadata;

% Before training on real dataset, we test our algorithm on a simple dataset 
x = [0,0;2,2;2,0;3,0];
y = [0;0;1;1];
c = [1;1;1;1];
x_hom = [c x]; % homogeneous form

%% write your own function for logistic regression as lr_yourname.m 
% input:
%      x_hom:    data matrix with homogeneous form
%       y:      label, a vector
%       option:  1-->GD, 2-->Newton, 3-->BFGS, 4-->modified BFGS
% output:
%      weight:  parameters in logistic regression weight = [b, w]
%      glist:   record the norm of gradient in iteration, a vector

[ weight_simp_1, glist_simp_1, J_rec_simp_1 ] = lr_zhangyedi(x_hom, y,1); % GD
[ weight_simp_2, glist_simp_2, J_rec_simp_2 ] = lr_zhangyedi(x_hom, y,2); % Newton
[ weight_simp_3, glist_simp_3, J_rec_simp_3 ] = lr_zhangyedi(x_hom, y,3); % BFGS

%% plot ||\nabla g|| of simple dataset
h1 = sigmoid( x_hom * weight_simp_1 );
h2 = sigmoid( x_hom * weight_simp_2 );
h3 = sigmoid( x_hom * weight_simp_3 );

subplot(1,3,1);
semilogy(glist_simp_1);
set(gca,'FontSize',15)
xlabel('iteration','FontSize',15)
ylabel('log ||\nabla g||', 'FontSize',15)
title(sprintf('GD'));

subplot(1,3,2);
semilogy(glist_simp_2);
set(gca,'FontSize',15)
xlabel('iteration','FontSize',15)
ylabel('log ||\nabla g||', 'FontSize',15)
title(sprintf('Newton'));

subplot(1,3,3);
semilogy(glist_simp_3);
set(gca,'FontSize',15)
xlabel('iteration','FontSize',15)
ylabel('log ||\nabla g||', 'FontSize',15)
title(sprintf('BFGS'));


nba_datahom = [ones(size(nba_data,1), 1) nba_data(:,1:end-1)];
cham_label = nba_data(:,end);

%% write your own function for logistic regression as lr_yourname.m

% ------ run BFGS on nbadata.mat ----------- %

[ weight3, glist3, J_rec3]=lr_zhangyedi(nba_datahom, cham_label,3);

% compute the accuracy of prediction in training set
h3 = sigmoid( nba_datahom * weight3 );
y_p3 = process_h(h3);
pred_accuracy3 = sum(abs(y_p3-cham_label));
fprintf('the number of misclassified data in BFGS is: %f, and pred_accuracy is: %f ',pred_accuracy3,1-pred_accuracy3/765);


% ------ run modified BFGS on nbadata.mat ----------- %

[ weight4, glist4, J_rec4]=lr_zhangyedi(nba_datahom, cham_label,4);

% compute the accuracy of prediction in training set
h4 = sigmoid( nba_datahom * weight4 );
y_p4 = process_h(h4);
pred_accuracy4 = sum(abs(y_p4-cham_label));
fprintf('\nthe number of misclassified data in modified BFGS is: %f, and pred_accuracy is: %f ',pred_accuracy4,1-pred_accuracy4/765);

% subplot(1,2,1);
% semilogy(glist3);
% set(gca,'FontSize',15)
% xlabel('iteration','FontSize',15)
% ylabel('log ||\nabla g||', 'FontSize',15)
% title(sprintf('BFGS on nba-data'));
% 
% subplot(1,2,2);
% semilogy(glist4);
% set(gca,'FontSize',15)
% xlabel('iteration','FontSize',15)
% ylabel('log ||\nabla g||', 'FontSize',15)
% title(sprintf('modified BFGS on nba-data'));


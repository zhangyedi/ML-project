clear;
sample=[1 1 -1; 1 -1 -1;-1 1 -1; -1 -1 -1;
    2 2 1;2 -2 1; -2 2 1; -2 -2 1];
cvx_begin    
       variables ome(3) b(1)
       minimize (sum_square(ome)/2)
       subject to
            sample(1,3)*(ome'*sample(1,:)'+b)>=1
            sample(2,3)*(ome'*sample(2,:)'+b)>=1
            sample(3,3)*(ome'*sample(3,:)'+b)>=1
            sample(4,3)*(ome'*sample(4,:)'+b)>=1
            sample(5,3)*(ome'*sample(5,:)'+b)>=1
            sample(6,3)*(ome'*sample(6,:)'+b)>=1
            sample(7,3)*(ome'*sample(7,:)'+b)>=1
            sample(8,3)*(ome'*sample(8,:)'+b)>=1
cvx_end

% plot the hyperplane %
ezplot('x^2+y^2 = 5');
hold on;
plot(sample(1:4,1),sample(1:4,2),'dr');
hold on;
plot(sample(5:8,1),sample(5:8,2),'*');




function [ weight, glist, J_rec] =lr_zhangyedi(x_hom,y,option)
% input:
%      x_hom:   data matrix with homogeneous form
%      y:       label, a vector
%      option:  1-->GD, 2-->Newton, 3-->BFGS, 4-->modified BFGS 
% output:
%      weight:  parameters in logistic regression weight = [b, w]
%      glist:   record the norm of gradient in iteration, a vector
%      J_rec:   record the loss 
    
    [m,n] = size(x_hom); % x_hom = 4*3 --sample
    initialWeight = zeros(n,1);    
    glist=[0];
    J_rec=[0];
    J=inf;
    accurancy=0.0001;
    max_iter=10000;
    if(option ==1 ) 
        iter=0;
        while(iter==0 || glist(iter+1) > accurancy)
            [J, grad, new_weight] = computeCost(initialWeight,x_hom,y,option);
            glist=[glist grad];
            J_rec=[J_rec J];
            initialWeight=new_weight;
            iter=iter+1;
            if(iter>max_iter)
                break;
            end
        end
        glist=glist';
        J_rec=J_rec';
    elseif(option==2)
        iter=0;
        while(iter==0 || glist(iter+1) > accurancy)
            [J, grad, new_weight] = computeCost(initialWeight,x_hom,y,option);
            glist=[glist grad];
            J_rec=[J_rec J];
            initialWeight=new_weight;
            iter=iter+1;
            if(iter>max_iter)
                break;
            end
        end
        glist=glist';
        J_rec=J_rec';
    elseif(option==3)
        iter=0;
        h = sigmoid( x_hom * initialWeight );
        g = x_hom'* (h-y);
        J = - sum(y .* log(h) + (1-y) .* log(1- h) );
        D=diag(h .*(1-h),0);
        H = x_hom'* D * x_hom;
        invH = inv(H);
        new_weight = initialWeight;
        
        while(iter==0 || glist(iter+1) > accurancy )
            new_weight = initialWeight-invH * g;
            h = sigmoid( x_hom * new_weight );
            new_g = x_hom'* (h-y);
            glist = [glist norm(new_g,2)];
            
            J = - sum(y .* log(h) + (1-y) .* log(1- h) );
            J_rec=[J_rec J];
            
            s = new_weight-initialWeight;  
            f = new_g-g;
            new_invH=(eye(n)- s*f'/(f'*s))*invH*(eye(n)-f*s'/(f'*s))+s*s'/(f'*s);
        
            g = new_g;
            invH = new_invH;
            initialWeight = new_weight;
            iter=iter+1;
            if(iter>max_iter)
                break;
            end
        end
        glist=glist';
        J_rec=J_rec';
    elseif(option==4)
        % modified BFGS
        iter=0;
        h = sigmoid( x_hom * initialWeight );
        g = x_hom'* (h-y);
        J = - sum(y .* log(h) + (1-y) .* log(1- h) );
        D=diag(h .*(1-h),0);
        H = x_hom'* D * x_hom;
        invH = inv(H);
        new_weight = initialWeight;
        while( iter==0 || glist(iter+1)>accurancy )
            new_weight = initialWeight-invH * g;
            h = sigmoid( x_hom * new_weight );
            % -------- modify part in BFGS begin ---------- %
             index_pos = find(h>=0.5);
             index_neg = find(h<0.5);
             h(index_pos) = h(index_pos)+2.0*(h(index_pos)-0.5).*(h(index_pos)-1);
             h(index_neg) = h(index_neg)+2.0*(h(index_neg)-0.5).*(h(index_neg));
            % -------- modify part in BFGS end ---------- %
            new_g = x_hom'* (h-y);
            glist = [glist norm(new_g,2)];
            J = - sum(y .* log(h) + (1-y) .* log(1- h) );
            J_rec=[J_rec J];
            
            s = new_weight-initialWeight;  
            f = new_g-g;
            new_invH=(eye(n)- s*f'/(f'*s))*invH*(eye(n)-f*s'/(f'*s))+s*s'/(f'*s);
        
            g = new_g;
            invH = new_invH;
            initialWeight = new_weight;
            iter=iter+1;
            if(iter>max_iter)
                break;
            end
        end
        glist=glist';
        J_rec=J_rec';
    end
     weight=initialWeight;
    
    
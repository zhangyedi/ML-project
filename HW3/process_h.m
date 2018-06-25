function new_y =process_h(h)
% ---------- recover y_predict ------------- %
% input:
%      h:      the value calculated from regression function
% output:
%      new_y:  recovered judgement
    new_h = h;
    for i=1:size(h,1)
        if(h(i)>=0.5)
            new_h(i)=1;
        else
            new_h(i)=0;
        end
    end
    
    new_y = new_h;

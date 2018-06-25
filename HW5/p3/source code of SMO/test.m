clear all;
[data test_data ] = get_a3a_data();
data=zscore(data);
accu_rec= [0];
pass=91;
count=1;
data_pass=data(1:pass*count,:);
while(1)
    count
    Accuracy = anothersmo(data_pass,test_data);
    accu_rec=[accu_rec Accuracy];
    count=count+1;
    if count==36
        break;
    end
    data_pass=data(1:pass*count,:);
end
x=0:1:35;
plot(x,accu_rec);
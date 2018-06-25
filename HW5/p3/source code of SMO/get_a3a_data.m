function [train_data test_data ]= get_a3a_data()
train_data=zeros(3185,124);
fid=fopen('a3a');
count=1;
while count<3185+1
    tline = fgetl(fid);
    tline=strrep(tline,':1',' ');
    train_data(count,1)=str2num(tline(1:2));
    data=str2num(tline(4:length(tline)));
    [m,n]=size(data);
    for i=1:n
        train_data(count,data(i))=1;
    end
    count=count+1;
end
fclose(fid);
test_data=zeros(29376,124);
fid=fopen('a3a.t');
count=1;
while count<29376+1
    tline = fgetl(fid);
    tline=strrep(tline,':1',' ');
    test_data(count,1)=str2num(tline(1:2));
    data=str2num(tline(4:length(tline)));
    [m,n]=size(data);
    for i=1:n
        test_data(count,data(i))=1;
    end
    count=count+1;
end
fclose(fid);

% fid=fopen('train_data','a');  
% [x,y]=size(train_data);  
% for i=1:x  
%     for j=1:y-1  
%         fprintf(fid,'%f\t',train_data(i,j));  
%     end  
%     fprintf(fid,'%f\n',train_data(i,y));%每一行回车\n  
% end  
% fclose(fid); 
% 
% fid=fopen('test_data','a');  
% [x,y]=size(test_data);  
% for i=1:x  
%     for j=1:y-1  
%         fprintf(fid,'%f\t',test_data(i,j));  
%     end  
%     fprintf(fid,'%f\n',test_data(i,y));%每一行回车\n  
% end  
% fclose(fid); 

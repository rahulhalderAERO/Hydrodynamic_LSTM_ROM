% HPC_NEW = zeros(3000,7);
% for i = 1 : 300
%     i
%      for k = 1:6000
%          if HPC(k,1)*100 <= 5*i
%               k_check = k;
%          end
%      end
%      k_check
%      HPC_NEW(i,:)= HPC(k_check,:);
% end
for i =1:601
    Exp_data(i,:)= Exp(5*i-4,:);
end

for i =1:601
    Exp_paddle(i,:)= paddle(2*i-1,:);
end





function r_out = SVM_RFE_tyb_fast(label, data, num_group, num_in_group)

% 当data为二维数组时
% num_in_group: 组内有几个特征 size（data，2）
% num_group：组数  size（data，1）
% r_out：组数的重要程度输出

    % SVM-RFE
    % SVM Recursive Feature Elimination (SVM RFE)
    % by liyang @BNU Math
    % Email:patrick.lee@foxmail.com
    % last modified 2010.09.18
    
%     data = randn(216,720);
%     label = randn(216,1)>0;
%     numclass=size(unique(label),1);
    numclass = 2;
    Num = size(data,2);%lyl特征的总数，即数据矩阵的列数
    
    
    s = [1:Num];%lyl 初始化一个特征索引向量，从 1 到 Num
    s_tmp = s;
    r = [];%lyl初始化一个空向量，用于存储被选中的特征索引 删除的索引在后，
    r_tmp = r;
    iter = 1;  %  删除的特征个数

    if numclass==2

        
        % grouped SVM
        while ~isempty(s)%lyl只要还有特征索引 s 不为空，就继续循环
            X = data(:,s);%lyl选择当前剩余的特征集合 s，构建特征矩阵 X
            % model = libsvmtrain(label, X);
            tmp =  size(s,2)/num_in_group - num_group;%lyl计算特征组需要删除的数 54/3-10
        
            if tmp>0    % 继续排序           
               model = fitcsvm(X,label);   %by tyb
               w = model.SupportVectors' * (model.SupportVectorLabels.*model.Alpha);%%求权重  nSV 是各个类的支持向量数量   在sv_coef中每一行存放该支持向量与其它各类（n-1类）进行一对一训练时系数
               c = w.^2;%lyl计算每个特征权重的平方，得到特征的重要性评分
               if num_in_group >1
                 c_r = reshape(c,length(c)/num_in_group,num_in_group);%lyl将权重向量 c 重新排列为矩阵 c_r，每行表示一个特征组，包含 num_in_group 个特征的权重
                 c_r_1 = sum(c_r');%lyl对每个特征组内的权重进行求和，得到每个组的综合权重 c_r_1
               else
                   c_r = c;
                   c_r_1 = c;
               end
%                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                [~ , f_1] = min(c_r_1);
%                f = f_1:size(c_r,1):length(c);
%                r = [s(f),r];
%                s(f) = [];    % 里面是绝对索引，妙！
%                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
               
               %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
               % 折半去除
               [~, c_indx] = sort(c_r_1,'ascend');% lyl对综合权重 c_r_1 进行升序排序，c_indx包含了 c_r_1 中元素按升序排序后的原始索引表示从最不重要到最重要的特征组顺序
               num_reduce = floor(max(tmp,2)/2);  % 需要减少的组数%lyl 这行代码的作用是：首先找到 tmp 和 2 中的较大值，然后将该值除以 2，最后向下取整得到一个整数。
%                num_reduce = 1;
               tmp_c = c_indx(1:num_reduce);  % 选出组数的索引,相对索引 % lyl选择权重最小的 num_reduce 个特征组的索引
               if num_in_group==1
%                    tmp_c = tmp_c';
               else
                   tmp_c = tmp_c'*ones(1,num_in_group) + ones(length(tmp_c),1)*[0:num_in_group-1]*size(c_r,1);% lyl将特征组索引转换为具体特征索引：生成需要移除的特征索引列表：
               end
               tmp_c = tmp_c';
               r = [s(tmp_c(:)),r];
               s(tmp_c(:)) = [];    % 里面是绝对索引，妙！
               %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
               
            else
               % 满足要求后，直接填充r 
               s_r = reshape(s,length(s)/num_in_group,num_in_group);%lyl 这里 s 被重塑为一个新的矩阵 s_r，行数为 length(s) / num_in_group，列数为 num_in_group
               s_1 = s_r';          % 按每隔3个为一组%lyl s_r' 是 s_r 的转置操作，即将 s_r 的行和列交换
               r = [s_1(:)' r];   
               %lyl s_1(:)将矩阵 s_1 展平为一个列向量，包含所有的元素
               %lyls_1(:)'是将 s_1(:) 转置为行向量（即将所有元素排列为一行）
               %lyl [s_1(:)' r] 是将 s_1(:)' 和原来的 r 拼接成一个新的数组 r，拼接的方向是将 s_1(:)' 插入到 r 的前面。    
               break;  
            end
        end

         r_out = r(1:num_in_group:end);  % r是3个一组！lyl将从 r 中提取每隔 num_in_group 个元素的值，从第一个元素开始，直到 r 的最后一个元素,前面10个数值没有排序，后面八个相关性较低。
         %lyl   r_out_10 = r_out(1:10); %lyl 读取前10个值 没有先后重要性排列
                

        end
%         display(iter);
    end
% end
function [train_h0_data, train_h0_edge_attr, train_h0_label,test_h0_edge_attr,train_h1_data, train_h1_edge_attr, train_h1_label,test_h1_edge_attr,test_h0_label, test_h1_label]=svm_two_class(index, data_name,numF)
% Include dependencies

addpath('./lib');                      % dependencies
addpath('./methods');            % FS methods
addpath(genpath('./lib/drtoolbox'));
addpath('./whole_brain'); % datasets

%data_name = 'Peking_data_m';%lyl  1/3
%index =1;%lyl  2/3

listFS = {'ILFS','InfFS','ECFS','mrmr','relieff','mutinffs','fsv','laplacian','mcfs','rfe','L0','fisher','UDFS','llcfs','cfs'};
selection_method = listFS{10};           % Selected rfe

%numF = 25;%lyl  3/3
% Load the data and select features for classification
% load fisheriris
load([data_name '.mat']);

X_temp_FC = inform_m.brain_conn_show;                 % FC
X_temp_ALFF =inform_m.ALFF;
X_temp_ALFF_cell =inform_m.ALFF_cell;
Y_temp = (inform_m.tag>0);          % ADHD 1,HC 0
Y = nominal(ismember(Y_temp,1)); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
brain_28 = [7:8, 11:14, 21:22, 29:42,63:66, 77:78];
brain_34 = [7:8, 11:16, 21:22, 25:26,29:42,63:66, 77:78,82,86];
brain_50 = [5:16,19:22, 25:46,63:66, 71:78];
for i=1:size(X_temp_ALFF,1)
          X_temp_ALFF_cell_1{1,i} = X_temp_ALFF_cell{1,i}(:,brain_50);
          X_temp_ALFF_cell_2{1,i} = reshape(X_temp_ALFF_cell{1,i}(:,brain_50)',[50*3,1]);
end
X_temp_ALFF_cell = X_temp_ALFF_cell_1;
X_temp_ALFF= cell2mat( X_temp_ALFF_cell_2)';

   
extractSubMatrix = @(matrix) matrix(brain_50, brain_50);

    % 使用 cellfun 应用于 cellArray 中每一项
X_temp_FC = cellfun(extractSubMatrix, X_temp_FC, 'UniformOutput', false);    


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[train_h0_data,train_h0_edge_attr,train_h0_label, test_h0_label,test_h0_edge_attr] = train_h0(index,X_temp_ALFF,X_temp_ALFF_cell,X_temp_FC,Y, selection_method, Y_temp, numF,data_name);

%[train_h0_data, train_h0_label] = energy_normalization(train_h0_data, train_h0_label);

[train_h1_data,train_h1_edge_attr,train_h1_label,test_h1_label,test_h1_edge_attr] = train_h1(index,X_temp_ALFF,X_temp_ALFF_cell,X_temp_FC,Y, selection_method, Y_temp, numF,data_name);

%[train_h1_data, train_h1_label] = energy_normalization(train_h1_data, train_h1_label);

end

function [train_data_out, train_label_out] = energy_normalization(train_data, train_label)

    tmp = train_data';
    sample_energy_tmp = sqrt(sum(tmp.^2));
    
    agv_energy_1 = mean(sample_energy_tmp(train_label));
    avg_energy_0 = mean(sample_energy_tmp(~train_label));
    sizeoftmp = size(tmp);
    sample_energy_map = ones(1, sizeoftmp(2));
    sample_energy_map(train_label) = agv_energy_1;
    sample_energy_map(~train_label) = avg_energy_0;
    
    energy_map = ones(size(tmp,1),1) * sample_energy_map;
    
    train_data_out = (tmp ./ energy_map)'; 
    train_label_out = train_label;
    
    return
end

function [train_h0_data,train_h0_edge_attr,train_h0_label,test_h0_label,test_h0_edge_attr] = train_h0(index,X_temp_ALFF,X_temp_ALFF_cell,X_temp_FC,Y, selection_method, Y_temp,numF,data_name)
X = X_temp_ALFF;
X_train = double(X);
Y_train = (double(Y)-1)*2-1; % labels: HC -1, ADHD +1   

X_test = double( X(index,:) );
Y_test = (double( Y(index) )-1)*2-1; % labels: neg_class -1, pos_class +1
test_h0_label = double(Y(index));

k = numF;

% feature Selection on training data
   switch lower(selection_method)
        case 'rfe'
            ranking = SVM_RFE_tyb_groupfast(Y_train, X_train, k, 3);
    end

%save([data_name '_rank25h050_FC_3alff.mat'], 'ranking');
%save([data_name '_rank50_FC_3alff.mat'], 'ranking');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     for i=1:size(X_train,1)
          X_temp_ALFF_cell{1,i} = reshape(X_temp_ALFF_cell{1,i}(:,ranking(1:k))',[k*3,1]);
     end
    X_train = cell2mat(X_temp_ALFF_cell)';

    %  train_h0_data中包含的测试数据
    %k=numF;
    train_h0_data = X_train;
   
    extractSubMatrix = @(matrix) matrix(ranking(1:k), ranking(1:k));

    % 使用 cellfun 应用于 cellArray 中每一项
    train_h0_edge_attr = cellfun(extractSubMatrix, X_temp_FC, 'UniformOutput', false);




    train_h0_label = Y_temp;
    % 删除当前索引的数据
    
    

    test_h0_edge_attr = train_h0_edge_attr(index);

    train_h0_data(index,:)=[];
    train_h0_label(index)=[];
    train_h0_edge_attr(index)=[];

    % 排序
    [~,indx_1] = sort(train_h0_label,'descend');
    train_h0_label= train_h0_label(indx_1);
    train_h0_data = train_h0_data(indx_1,:);
    train_h0_edge_attr =  train_h0_edge_attr(1,indx_1);
   
   
    
    return
end

function [train_h1_data,train_h1_edge_attr,train_h1_label,test_h1_label,test_h1_edge_attr] = train_h1(index,X_temp_ALFF,X_temp_ALFF_cell,X_temp_FC,Y, selection_method, Y_temp,numF,data_name)
X = X_temp_ALFF;
X_train = double(X);
Y_temp(index) = ~Y_temp(index);
if Y(index) == 'true'
    Y(index,1) = 'false';
else
    Y(index,1) = 'true';
end

Y_train = (double(Y)-1)*2-1; % labels: neg_class -1, pos_class +1

X_test = double( X(index,:) );
Y_test = (double( Y(index) )-1)*2-1; % labels: neg_class -1, pos_class +1
test_h1_label = double(Y(index));
k = numF; 
% feature Selection on training data
    switch lower(selection_method)
        case 'rfe'
            ranking = SVM_RFE_tyb_groupfast(Y_train, X_train, k, 3);
    end
%save([data_name '_rank25h150_FC_3alff.mat'], 'ranking');
        
     for i=1:size(X_train,1)
          X_temp_ALFF_cell{1,i} = reshape(X_temp_ALFF_cell{1,i}(:,ranking(1:k))',[k*3,1]);
     end
    X_train = cell2mat(X_temp_ALFF_cell)';
    
    
    train_h1_data = X_train;
   
    extractSubMatrix = @(matrix) matrix(ranking(1:k), ranking(1:k));

     % 使用 cellfun 应用于 cellArray 中每一项
    train_h1_edge_attr = cellfun(extractSubMatrix, X_temp_FC, 'UniformOutput', false);
    
    train_h1_label = Y_temp;


    

    test_h1_edge_attr = train_h1_edge_attr(index);

     % 删除当前索引的数据
    train_h1_data(index,:)=[];
    train_h1_label(index)=[];
    train_h1_edge_attr(index)=[];
    
    % 排序
    [~,indx_1] = sort(train_h1_label,'descend');
    train_h1_label= train_h1_label(indx_1);
    train_h1_data = train_h1_data(indx_1,:);
    train_h1_edge_attr =  train_h1_edge_attr(1,indx_1);
   
    
    return
end
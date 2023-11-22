clc;
clear
tic;
format long;
format compact;

algorithmDir = 'SHADE_DNM';
imgPath = ['ROC'];
mkdir(imgPath);
max_run_times = 1;%30

%% dataset list
%problem_list = ["Ionosphere","Australia","KrVsKpEW","BreastEW","German","Heart","CongressEW","SpectEW","Exactly","Tic-tac-toe","Vote"]
problem_list = ["Tic-tac-toe","Vote"];
problem_list_len = length(problem_list);

%% Hyper-parameters
% Read from the read_data.m by default, you can also modify it yourself 
M_list  = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
qs_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
k_list  = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

%% The parameter of SHADE
popsize = 100; % the size of population
FES = 30000;

times = [];

%% start loop
for problem_i= 1:problem_list_len
    t=clock;
    problem_name = char(problem_list(problem_i));
    
    disp(['The current running problem: ',problem_name]);

    [train_data,train_target,test_data,test_target,M,k,qs] = read_data(problem_name);
    % M  = get_hp(M_list, problem_i, problem_list_len, M);
% M=10;
    qs = get_hp(qs_list, problem_i, problem_list_len, qs);
    k  = get_hp(k_list, problem_i, problem_list_len, k);
    
    if M == 0 || qs == 0 || k == 0
        disp("Please check the Hyper-parameters! Got a Hyper-parameters with a value of 0 ")
        return 
    end 
    disp(['M:',num2str(M),'  k:',num2str(k),'  qs:',num2str(qs)])
    
    [I,J] = size(train_data);
    train_len = length(train_target);
    test_len = length(test_target);

    %% parameter structure
    Parameter.popsize = popsize;
    Parameter.FES = FES;
    D = J * M * 2; % the dimension of sample
    Parameter.D = D;
    Parameter.problem_name = problem_name;
    net.M = M;
    net.qs = qs;
    net.k = k;    
    
    result = [];
    test_fit_Chart = [];
    analysisChart = [];
    resultChart = [];
    optimalChart = [];
    Accuracy = [];
    AUC = [];
    Sensitivity = [];
    Specificity = [];
    kqsM=[];
    
    for runtimes = 1:max_run_times
        %% train
        [w,q,qs,k,best,out] =trainSHADE(Parameter,net,train_data,train_target);
        net.w=w;
        net.q=q;
        net.qs=qs;
        net.k=k;
        %% test
        test_fit=my_DNM(test_data,net);
        test_fit=round(test_fit);
        count=length(find(test_fit == test_target));
        Accuracy = [Accuracy, count / test_len];
        % out{runtimes}.testacc = Accuracy(end);
        writematrix(out.convergence,['HSHADE_' num2str(problem_name) '.xlsx']); out.convergence
        %% AUC & sensitivity & specificity
        [auc, sensitivity, specificity] = plot_roc(test_fit, test_target, imgPath, algorithmDir, problem_name, runtimes);
        AUC = [AUC, auc];
        Sensitivity = [Sensitivity, sensitivity];
        Specificity = [Specificity, specificity];
        AUC_Sensitivity_Specificity = [AUC; Sensitivity; Specificity];
            
        % result = [result,best];
        % resultChart = [resultChart,out{runtimes}.convergence];%每次??代匮值的列??
        % resultChart = cat(1,resultChart,out{runtimes}.convergence);
        test_fit_Chart = [test_fit_Chart, test_fit];
        kqsm=[k;qs;M];
        kqsM=[kqsM kqsm];
        
        %% write w and q in xls
        % path_w_q = ['./',algorithmDir,'_',problem_name,'_D',num2str(D),'_w_and_q'];
        % xlswrite(path_w_q,w,['w_time',num2str(runtimes)]);
        % xlswrite(path_w_q,q,['q_time',num2str(runtimes)]);
    end
    
    PF_Time = etime(clock,t);
    times = [times, PF_Time];
    fprintf('\t\t 计算耗时：     %f  秒\n\n',PF_Time);
    
    % save   SHADE_Vote.mat out
    % resultChart = resultChart.'
    % xlswrite(['kqsM','_',problem_name],kqsM,'test_adv');
    
    mean_Accuracy = mean(Accuracy,2);
    std_Accuracy = std(Accuracy,0,2);
    Accuracy = [Accuracy, mean_Accuracy];
    Accuracy = [Accuracy, std_Accuracy];
    path_accuracy = ['./',algorithmDir,'_',problem_name,'_D',num2str(D),'_accuracy'];
    % xlswrite(path_accuracy,Accuracy,['accuracy']);

    % mean_AUC_Sensitivity_Specificity = mean(AUC_Sensitivity_Specificity,2);
    % std_AUC_Sensitivity_Specificity = std(AUC_Sensitivity_Specificity,0,2);
    % AUC_Sensitivity_Specificity = [AUC_Sensitivity_Specificity, mean_AUC_Sensitivity_Specificity];
    % AUC_Sensitivity_Specificity = [AUC_Sensitivity_Specificity, std_AUC_Sensitivity_Specificity];
    % 
    % path_AUC_Sensitivity_Specificity = ['./',algorithmDir,'_',problem_name,'_D',num2str(D),'_AUC_Sensitivity_Specificity'];
    % xlswrite(path_AUC_Sensitivity_Specificity,AUC_Sensitivity_Specificity,['AUC_Sen_Spe']);
    % 
    % path = ['./',algorithmDir,'_',problem_name,'_D',num2str(D),'_error'];
    % xlswrite(path,resultChart,['error']);
    % 
    % path_test_fit = ['./',algorithmDir,'_',problem_name,'_D',num2str(D),'_test_fit'];
    % xlswrite(path_test_fit,test_fit_Chart,['test_fit']);
    % 
    % analysisChart = [analysisChart;resultChart(end,:)];
    % mean_ = mean(analysisChart,2);
    % std_ = std(analysisChart,0,2);
    % analysisChart = [analysisChart,mean_];
    % analysisChart = [analysisChart,std_];
    % xlswrite(path,analysisChart,['Statistic']);
    
end
% times = [times, sum(times)];
% problem_list2 = [problem_list, "总计" ];
% save_time = [problem_list2; times];
% path_time = ['./',algorithmDir,'_time'];
% xlswrite(path_time,save_time,['times']);
toc;

function [auc, sensitivity, specificity] = plot_roc(deci, label_y, imgPath, algorithmDir, problem_name, runtimes) 
    [val,ind] = sort(deci,'descend');
    roc_y = label_y(ind);
    stack_x = cumsum(roc_y == 0)/sum(roc_y == 0);
    stack_y = cumsum(roc_y == 1)/sum(roc_y == 1);
    auc = sum((stack_x(2:length(roc_y),1)-stack_x(1:length(roc_y)-1,1)).*stack_y(2:length(roc_y),1));
    plot(stack_x,stack_y);
    xlabel('False Positive Rate');
    ylabel('True Positive Rate');
    title(['ROC curve of (AUC = ' num2str(auc) ' )']);
    TP = sum((roc_y==1)&(val==1));
    TN = sum((roc_y==0)&(val==0));
    FN = sum((roc_y==1)&(val==0));
    FP = sum((roc_y==0)&(val==1));
    sensitivity = TP / (TP + FN);
    specificity = TN / (TN + FP);
    % saveas(gcf,[[imgPath '/'],algorithmDir,'_',problem_name,'time',num2str(runtimes) '.png']);
    % saveas(gcf,[[imgPath '/'],algorithmDir,'_',problem_name,'time',num2str(runtimes) '.fig']);
end

function [r] = get_hp(list, i, pl_len, dr)
    if length(list) < pl_len
        r = 0;
    else
        r = list(i);
    end
    if r == 0
        r = dr;
    end
end
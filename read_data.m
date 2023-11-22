function [train_data,train_target,test_data,test_target,M,k,qs] = read_data(problem_name)

addpath('C:\Users\user09\Desktop\DNM\DNM\Dataset');

M = 0;
qs = 0;
k = 0;

switch problem_name
    %% classification
    case 'SpectEW'
        [train_data, train_target, test_data, test_target] = load_data_type1('SpectEW_data.mat');
        M = 1;
        qs = 0.1;
        k = 5;
        
    case 'CongressEW'
        [train_data, train_target, test_data, test_target] = load_data_type1('CongressEW_data.mat');
        M = 3;
        qs = 0.1;
        k = 20;
        
    case 'Ionosphere'
        [train_data, train_target, test_data, test_target] = load_data_type1('IonosphereEW_data.mat');
        M = 1;
        qs = 0.5;
        k = 5;
        
    case 'Australia'
        [train_data, train_target, test_data, test_target] = load_data_type1('Australia_data.mat');
        M = 1;
        qs = 0.3;
        k = 20;
        
    case 'KrVsKpEW'
        [train_data, train_target, test_data, test_target] = load_data_type1('KrVsKpEW_data.mat');
        M = 1;
        qs = 0.1;
        k = 5;
        
    case 'BreastEW'
        [train_data, train_target, test_data, test_target] = load_data_type1('BreastEW_data.mat');
        M = 16;
        qs = 0.5;
        k = 5;
        
    case 'German'
        [train_data, train_target, test_data, test_target] = load_data_type1('German_data.mat');
        M = 1;
        qs = 0.3;
        k = 10;

    case 'Heart'
        load('Heart297.mat','heart297');
        train_data = heart297(:,1:10);
        train_target = heart297(:,11);
        test_data = heart297(:,1:10);
        test_target = heart297(:,11);
        M = 8;
        qs = 0.3;
        k = 10;
                
    case 'Exactly'
        [train_data, train_target, test_data, test_target] = load_data_type1('Exactly_data.mat');
        M = 10;
        qs = 0.9;
        k = 5;
        
    case 'Vote'
        [train_data, train_target, test_data, test_target] = load_data_type1('Vote_data.mat');
        M = 2;
        qs = 0.3;
        k = 15;
        
    case 'Tic-tac-toe'
        [train_data, train_target, test_data, test_target] = load_data_type1('Tic-tac-toe_data.mat'); 
        M = 9;
        qs = 0.9;
        k = 5;
        
    case 'Madelon'
        [train_data, train_target, test_data, test_target] = load_data_type1('Madelon_data.mat'); 
    case 'Sonar'
        [train_data, train_target, test_data, test_target] = load_data_type1('SonarEW_data.mat');
    case 'Musk'
        [train_data, train_target, test_data, test_target] = load_data_type1('Musk_data.mat');
    case 'Hillvalley'
        [train_data, train_target, test_data, test_target] = load_data_type1('Hillvalley_data.mat');
        
end
end


function [train_data, train_target, test_data, test_target] = load_data_type1(file_path)
    load(file_path,'data');
    train_data = data.train;
    train_target = data.trainLabel(:,1);
    test_data = data.test;
    test_target = data.testLabel(:,1);
end
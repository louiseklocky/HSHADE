function [w,q,qs,k,best,out] = trainSHADE(Parameter,net,train_data,train_target)
problem_size= Parameter.D;
pop_size = Parameter.popsize;
memory_size = pop_size;
max_nfes = Parameter.FES;
[~,J] =size(train_data); % J: the dimension of samples
lu = [-1 * ones(1, problem_size) 0 0; 1 * ones(1, problem_size) 1 * ones(1,1) 1 * ones(1,1)];
archive=[];
p_best_rate = 0.2;
arc_rate = 1;
archive.NP = arc_rate * pop_size; % the maximum size of the archive
archive.pop = zeros(0, problem_size); % the solutions stored in te archive
archive.funvalues = zeros(0, 1); % the function value of the archived solutions
M= net.M;

memory_sf = 0.5 .* ones(memory_size, 1);
memory_cr = 0.5 .* ones(memory_size, 1);
memory_pos = 1;
run_funcvals = [];

%% Initialize the main population
popold = repmat(lu(1, :), pop_size, 1) + rand(pop_size, problem_size+2) .* (repmat(lu(2, :) - lu(1, :), pop_size, 1));
pop = popold; % the old population becomes the current population
for popindex = 1:pop_size
    w = pop(popindex,1:J*M);
    q = pop(popindex,J*M+1:problem_size);
    w=reshape(w,J,M);
    q=reshape(q,J,M);
    net.w = w;
    net.q = q;
    net.qs = pop(problem_size+2);
    net.k = 10*pop(problem_size+1);
    train_fit = my_DNM(train_data,net);
    cost = (train_fit - train_target).^2;
    fitness(popindex,1) = mean(cost);
end
nfes = pop_size;
bsf_fit_var = min(fitness);
run_funcvals = [run_funcvals;bsf_fit_var];

%% main loop
while nfes < max_nfes
    disp(['nfes=',num2str(nfes)]);
    pop = popold; % the old population becomes the current population
    [~, sorted_index] = sort(fitness, 'ascend');

    mem_rand_index = ceil(memory_size * rand(pop_size, 1));
    mu_sf = memory_sf(mem_rand_index);
    mu_cr = memory_cr(mem_rand_index);
                
        %% for generating crossover rate
    cr = normrnd(mu_cr, 0.1);
    term_pos = find(mu_cr == -1);
    cr(term_pos) = 0;
    cr = min(cr, 1);
    cr = max(cr, 0);
        %% for generating scaling factor
    sf = mu_sf + 0.1 * tan(pi * (rand(pop_size, 1) - 0.5));
    pos = find(sf <= 0);
        
    while ~ isempty(pos)
        sf(pos) = mu_sf(pos) + 0.1 * tan(pi * (rand(length(pos), 1) - 0.5));
        pos = find(sf <= 0);
    end
        
    sf = min(sf, 1);

    %%

    r0 = [1 : pop_size];
        
    popAll = [pop; archive.pop];
    [r11, r22] = gnR1R2(pop_size, size(popAll, 1), r0);
        
    pNP = max(round(p_best_rate * pop_size), 2); %% choose at least two best solutions
    randindex = ceil(rand(1, pop_size) .* pNP); %% select from [1, 2, 3, ..., pNP]
    randindex = max(1, randindex); %% to avoid the problem that rand = 0 and thus ceil(rand) = 0
    pbest = pop(sorted_index(randindex), :); %% randomly choose one of the top 100p% solutions
    vi = pop + sf(:, ones(1, problem_size+2)) .* (pbest - pop + pop(r11, :) - popAll(r22, :));
        
    vi = boundConstraint(vi, pop, lu);
        
    mask = rand(pop_size, problem_size+2) > cr(:, ones(1, problem_size+2)); % mask is used to indicate which elements of ui comes from the parent
    rows = (1 : pop_size)'; cols = floor(rand(pop_size, 1) * (problem_size+2))+1; % choose one position where the element of ui doesn't come from the parent
    jrand = sub2ind([pop_size problem_size+2], rows, cols); mask(jrand) = false;
    ui = vi;
    ui(mask) = pop(mask);
    for index = 1:size(ui,1)
        w =ui(index,1:J*M);
        q =ui(index,J*M+1:problem_size);
        w=reshape(w,J,M);
        q=reshape(q,J,M);
        net.w = w;
        net.q = q;
        net.qs = ui(index,problem_size+2);
        net.k = 10*ui(index,problem_size+1);
        train_fit = my_DNM(train_data,net);
        cost = (train_fit - train_target).^2;
        children_fitness(index,1) = mean(cost);
    end
        
    for i = 1 : pop_size
        nfes = nfes + 1;
        if children_fitness(i) < bsf_fit_var
            bsf_fit_var = children_fitness(i);
        end
        if nfes > max_nfes; break; end
    end
        
    dif = abs(fitness - children_fitness);
        
        %% I == 1: the parent is better; I == 2: the offspring is better
    Child_is_better_index = (fitness > children_fitness);
    goodCR = cr(Child_is_better_index == 1);
    goodF = sf(Child_is_better_index == 1);
    dif_val = dif(Child_is_better_index == 1);
        
    archive = updateArchive_pin(archive, popold(Child_is_better_index == 1, :), fitness(Child_is_better_index == 1));

    [fitness, Child_is_better_index] = min([fitness, children_fitness], [], 2);
    popold = pop;
    popold(Child_is_better_index == 2, :) = ui(Child_is_better_index == 2, :);
        
    num_success_params = numel(goodCR);
        
    if num_success_params > 0
        sum_dif = sum(dif_val);
        dif_val = dif_val / sum_dif;
            
            %% for updating the memory of scaling factor
        memory_sf(memory_pos) = (dif_val' * (goodF .^ 2)) / (dif_val' * goodF);
            
            %% for updating the memory of crossover rate
        if max(goodCR) == 0 || memory_cr(memory_pos)  == -1
            memory_cr(memory_pos)  = -1;
        else
            memory_cr(memory_pos) = (dif_val' * (goodCR .^ 2)) / (dif_val' * goodCR);
        end
            
        memory_pos = memory_pos + 1;
        if memory_pos > memory_size;  memory_pos = 1; end
    end
        
    run_funcvals = [run_funcvals;bsf_fit_var];
        
end

[~,index] = min(fitness);
best_population = pop(index,:);
w = best_population(1:J*M);
q = best_population(J*M+1:problem_size);
w=reshape(w,J,M);
q=reshape(q,J,M);
out.convergence= run_funcvals;
out.w=w;
out.q=q;
qs = best_population(problem_size+2);
k = 10*best_population(problem_size+1);
best = bsf_fit_var;  

end %% end 1 ru


function archive = updateArchive_pin(archive, pop, funvalue)
% Update the archive with input solutions
%   Step 1: Add new solution to the archive
%   Step 2: Remove duplicate elements
%   Step 3: If necessary, randomly remove some solutions to maintain the archive size
%
% Version: 1.1   Date: 2008/04/02
% Written by Jingqiao Zhang (jingqiao@gmail.com)

if archive.NP == 0, return; end

if size(pop, 1) ~= size(funvalue,1), error('check it'); end

% Method 2: Remove duplicate elements
popAll = [archive.pop; pop ];
funvalues = [archive.funvalues; funvalue ];
[dummy IX]= unique(popAll, 'rows');
if length(IX) < size(popAll, 1) % There exist some duplicate solutions
  popAll = popAll(IX, :);
  funvalues = funvalues(IX, :);
end

if size(popAll, 1) <= archive.NP   % add all new individuals
  archive.pop = popAll;
  archive.funvalues = funvalues;
else                % randomly remove some solutions
  rndpos = randperm(size(popAll, 1)); % equivelent to "randperm";
  rndpos = rndpos(1 : archive.NP);
  
  archive.pop = popAll(rndpos, :);
  archive.funvalues = funvalues(rndpos, :);
end
end

% Carrega os datasets.
Xraw = csvread('X.csv');
Yraw = csvread('Y.csv');

% Configura o gerador aleatório para uma semente fixa.
rng('default');
rng(1);

% Define o intervalo de número de neurônios a ser investigado.
% Cria variáveis para armazenar os mínimos, 
% máximos e médias das taxas de acerto.
range_ = 0:10:400;
min_   = zeros(1, length(range_));
max_   = zeros(1, length(range_));
mean_  = zeros(1, length(range_));
min_total   = zeros(1, length(range_));
max_total   = zeros(1, length(range_));
mean_total  = zeros(1, length(range_));
i = 1;

% Se o intervalo definido possuir mais de um valor, 
% salvar os dados em um arquivo, caso contrário, 
% apenas exibir no terminal.
savedata = false;
if length(range_) > 1
    % Define o número mínimo de neurônios para 5.
    range_(1) = 5;
    savedata = true;
end

% Para todos os valores do intervalo.
for n = range_

    n

    % Calcula as taxas de acerto com K-fold da RBF
    % com n neurônios na camada oculta.
    [success, total] = rbf10fold(Xraw, Yraw, n);

    % Calcula mínimos, máximos e médias
    min_(i)  = min(success);
    max_(i)  = max(success);
    mean_(i) = mean(success);
    min_total(i)  = min(total);
    max_total(i)  = max(total);
    mean_total(i) = mean(total);

    i = i + 1;

    % Salva ou apenas imprime os dados no terminal, 
    % de acordo com a variável savedata.
    if savedata
        data = [range_; min_; max_; mean_; min_total; max_total; max_total];
        csvwrite('output/rbfdata.csv', data);
    else
        n
        success
        min(success)
        max(success)
        mean(success)
        total
        min(total)
        max(total)
        mean(total)
    end

end
function [success, total] = rbf10fold(Xraw, Yraw, maxneurons)
    % Função rbf10fold
    % Avalia um modelo RBF com neurônios definidos pela variável maxneurons,
    % dados de entrada Xraw e rótulos Yraw com validação 10-fold.
    % Retorna o vetor success, contendo as taxas de acerto para cada fold.

    % Variáveis calculadas a partir dos dados de entrada e rótulos
    % Respectivamente, tamanho da camada de entrada, tamanho da camada 
    % de saída, número total de amostras, valor K usado no K-fold e 
    % tamanho de cada fold.
    inputsize = size(Xraw,1);
    outputsize = size(unique(Yraw),1);
    samples = size(Xraw,2);
    K = 10;
    foldsize = samples/K;

    % Os valores brutos dos dados de entrada já estão ajustados.
    X = Xraw;

    % Os valores brutos dos rótulos estão numerados de 0 a 29.
    % Este trecho de código primeiramente numera de 1 a 30.
    % Depois altera a representação da classe para a codificação one-hot.
    Yraw = Yraw + 1;
    Yraw = Yraw';
    Y = zeros(outputsize, samples);
    for i = 1:samples
        Y(Yraw(1,i),i) = 1;
    end

    % Variáveis para armazenamento dos K folds.
    % KX armazena os dados de entrada separados em K folds.
    % KY armazena os dados de saída separados em K folds.
    % h armazena a quantidade de itens em cada fold.
    KX = zeros(K,inputsize,foldsize);
    KY = zeros(K,outputsize,foldsize);
    h =  zeros(1,K);

    % Este trecho de código utiliza a função crossvalind para selecionar
    % randomicamente um fold para cada amostra. Essa seleção ocorre 
    % separadamente para cada classe, garantindo assim que cada fold 
    % contenha o mesmo número de amostras de cada classe.
    indices = [];
    for i = 1:outputsize
        indices = [indices; crossvalind('Kfold', samples/outputsize, K)];
    end

    % Assinala cada amostra ao seu fold selecionado previamente.
    for i = 1:samples
        h(indices(i)) = h(indices(i)) + 1;
        KX(indices(i),:,h(indices(i))) = X(:,i);
        KY(indices(i),:,h(indices(i))) = Y(:,i);
    end

    % Vetor para armazenar as taxas de acerto de cada um dos K folds.
    success = zeros(K,1);
    % Vetor para armazenar as taxas de acerto sobre todo o conjunto.
    total = zeros(K,1);

    % Realiza K rodadas de treinamento e teste, uma rodada para cada fold;
    for i = 1:K
        % Separa e ajusta as dimensões do conjunto de teste
        Xtest = KX(i,:,:);
        Ytest = KY(i,:,:);
        Xtest = reshape(Xtest,inputsize,foldsize);
        Ytest = reshape(Ytest,outputsize,foldsize);

        % Separa e ajusta as dimensões do conjunto de treinamento
        Xtrain = [];
        Ytrain = [];
        for j = 1:K
            if j ~= i
                xx = reshape(KX(j,:,:),inputsize,foldsize);
                yy = reshape(KY(j,:,:),outputsize,foldsize);
                Xtrain = [Xtrain xx];
                Ytrain = [Ytrain yy];
            end
        end

        % Configura e treina a RBF
        % Utiliza-se a função evalc para capturar o output de texto da
        % função newrb na variável T e assim desativar a impressão de 
        % texto no terminal MATLAB.
        [T, net] = evalc('newrb(Xtrain, Ytrain, 0, 2.0, maxneurons, maxneurons+10)');

        % Testa a RBF com o fold de teste. Em seguida, seleciona os neurônios
        % da camada de saída que tiveram a maior ativação e verifica se estes 
        % correspondem à classe das amostras. Caso positivo, registra acerto.
        y = net(Xtest);
        [m1,r1] = max(y);
        [m2,r2] = max(Ytest);
        result = (r1==r2);
        success(i) = sum(result)/foldsize;

        % Grava a matriz de confusao do fold em um arquivo csv caso 
        % o modelo seja RBF com uma camada oculta e 160 neuronios

        if maxneurons == 160
            confusionmatrix = zeros(outputsize);
            for j = 1:length(r1)
                confusionmatrix(r2(j), r1(j)) = confusionmatrix(r2(j), r1(j)) + 1;
            end
            csvwrite(sprintf('output/confmatrix-rbf-160-kfold-%d.csv', i), confusionmatrix);
        end

        % Testa a RBF com todo o dataset. Em seguida, seleciona os neurônios
        % da camada de saída que tiveram a maior ativação e verifica se
        % estes correspondem à classe correta das amostras.
        % Caso positivo, registra acerto.
        y = net(X);
        [m1,r1] = max(y);
        [m2,r2] = max(Y);
        result = (r1==r2);
        total(i) = sum(result)/samples;

        % Grava a matriz de confusao total em um arquivo csv caso 
        % o modelo seja RBF com uma camada oculta e 160 neuronios

        if maxneurons == 160
            confusionmatrix = zeros(outputsize);
            for j = 1:length(r1)
                confusionmatrix(r2(j), r1(j)) = confusionmatrix(r2(j), r1(j)) + 1;
            end
            csvwrite(sprintf('output/confmatrix-rbf-160-total-%d.csv', i), confusionmatrix);
        end

    end

    % Grava as taxas de acuracia total e do fold em um arquivo csv caso 
    % o modelo seja RBF com uma camada oculta e 160 neuronios
    if maxneurons == 160
        csvwrite('output/success-rbf-160.csv', success);
        csvwrite('output/total-rbf-160.csv', total);
    end

end
function [success, total] = mlp10fold(Xraw, Yraw, layers)
    % Função mlp10fold: avalia um modelo MLP com camadas ocultas 
    % definidas pela variável layers, dados de entrada Xraw e 
    % rótulos Yraw com validação 10-fold. Retorna o vetor success, 
    % contendo as taxas de acerto para cada fold.

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
    % Depois altera representação da classe para codificação one-hot.
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

    % armazena taxas de acerto de cada um dos K folds.
    success = zeros(K,1);
    % armazena taxas de acerto sobre todo o conjunto.
    total = zeros(K,1);

    % Realiza K rodadas de treinamento/ teste, uma para cada fold.
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

        % Configura a MLP
        net = feedforwardnet(layers);
        net.divideParam.trainRatio = 1;
        net.divideParam.valRatio   = 0;
        net.divideParam.testRatio  = 0;
        net.trainFcn = 'trainscg';
        net.trainParam.goal = 0.001;
        net.trainParam.epochs = 1000;
        net.trainParam.min_grad = 0;
        net.trainParam.showWindow = false;

        % Treina a MLP com os folds de treinamento
        net = train(net, Xtrain, Ytrain);

        % Testa a MLP com o fold de teste. Em seguida, seleciona os neurônios
        % da camada de saída que tiveram a maior ativação e verifica se estes 
        % correspondem à classe das amostras. Caso positivo, registra acerto.
        y = net(Xtest);
        [m1,r1] = max(y);
        [m2,r2] = max(Ytest);
        result = (r1==r2);
        success(i) = sum(result)/foldsize;

        % Grava a matriz de confusao do fold em um arquivo csv caso 
        % o modelo seja MLP com uma camada oculta e 200 neuronios

        if layers == [200]
            confusionmatrix = zeros(outputsize);
            for j = 1:length(r1)
                confusionmatrix(r2(j), r1(j)) = confusionmatrix(r2(j), r1(j)) + 1;
            end
            csvwrite(sprintf('output/confmatrix-mlp-200-kfold-%d.csv', i), confusionmatrix);
        end

        % Testa a MLP com todo o dataset. Em seguida, seleciona os neurônios
        % da camada de saída que tiveram a maior ativação e verifica se
        % estes correspondem à classe correta das amostras.
        % Caso positivo, registra acerto.
        y = net(X);
        [m1,r1] = max(y);
        [m2,r2] = max(Y);
        result = (r1==r2);
        total(i) = sum(result)/samples;

        % Grava a matriz de confusao total em um arquivo csv caso 
        % o modelo seja MLP com uma camada oculta e 200 neuronios
        if layers == [200]
            confusionmatrix = zeros(outputsize);
            for j = 1:length(r1)
                confusionmatrix(r2(j), r1(j)) = confusionmatrix(r2(j), r1(j)) + 1;
            end
            csvwrite(sprintf('output/confmatrix-mlp-200-total-%d.csv', i), confusionmatrix);
        end
    end

    % Grava as taxas de acuracia do fold e totais caso o modelo
    % seja MLP com uma camada oculta de 200 neuronios
    if layers == [200]
        csvwrite('output/success-mlp-200.csv', success);
        csvwrite('output/total-mlp-200.csv', total);
    end
end
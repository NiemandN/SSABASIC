classdef ssaBasic
    %SSABASIC performs a basic version of Singula Spectrum Analysis
    %   This class implements the Singular Spectrum Analysis (SSA)
    %   according to the contents of the book "Analysis of Time Series
    %   Structure: SSA and Related Techniques", N. Golyandina,
    %   V. Nekrutkin, and A. Zhigljavsky, 2001.
    %   From the introduction: SSA is essentially a model-free technique;
    %   it is more an exploratory, modelbuilding technique than a confirmatory
    %   procedure. It aims at a decomposition of the original series into
    %   a sum of a small number of interpretable components such as a
    %   slowly varying trend, oscillatory components and a structureless
    %   noise. The main concept in studying the SSA properties is
    %   separability, which characterizes how well different components
    %   can be separated from each other.
    %   Basic SSA analysis consists of four steps:
    %   1) Embedding
    %   2) Singular Value Decomposition (this and the previous step are
    %   performed within the class contructor)
    %   3) Grouping (this step is performed by the grouping method)
    %   4) Diagonal Averaging (this step is performed by the method
    %   hankelization)
    %   The previous steps lay the groundwork for capturing 
    %   the data generation process through the reconstruction method.
    %   Finally, the forecast is made using the forecast method,
    %   which applies a linear recursive formula.
    %   Diagnostic methods included in this class are:
    %   wcorrelation: weighted correlation to assess how separate are
    %   groups;
    %   plotSingularValue: scree plot to identify the leading singular
    %   values;
    %   scatterplotseigenvectors: scatter plot of the eigenvectors to
    %   capture the periodicy of their respective principal component;
    %   crossval_r0 and crossval_L0: respectively, the cross validation of
    %   the number of eigen-triples needed for signal reconstruction and
    %   the number of lags necessary to single out the relevant signal
    %   components (eigen-triples).
    %   Validation methods included in this class are:
    %   validateG0: ensures G0 is a non-empty, positive, numeric and without
    %   gaps array or scalar. Moreover, max(G0) < L0 + 1 and
    %   length(G0) < L0 + 1;
    %   validateL0: ensures L0 is a positive scalar less than half the
    %   number of observation;
    %   validateR0: ensures r0 is a non-empty, positive, numeric array or
    %   scalar and the max(r0) < L0 + 1;
    %   validateNumVal: ensures NumVal is a positive scalar less then or
    %   equal to the embedding dimension.
    %   backtest: evaluate the SSA forecast robustness by splitting the 
    %   data sample into in-sample and out-of-sample sets. Perform the forecast 
    %   on the in-sample data and compare the obtained forecast with the
    %   known data from the out-of-sample set.

    properties
        L  % Number of lags considered in the analysis
        N  % sample (x) dimension
        x  % Signal array
        U  % left eignevectors
        S  % singular values
        V  % right eigenvectors
        H  % Hankel matrix
        mX % mean of original signal
    end

    methods
        
        function obj = ssaBasic(x,L0)
            %SSABASIC Constructs an instance of this class
            %   obj = ssaBasic(x,L0) requires an array x and a positive
            %   scalar L0, representing the Lags for the analysis
            
            % check the number of input arguments
            if nargin < 1
                error(sprintf(strcat('Insufficient number of inputs.', ...
                'The function ssaBasic requires:\n\n - ', ...
                'A vector of observations (x) as the first input.\n - ', ...
                'An embedding dimension (L) as the second input.')))
            end

            % check x is an array
            obj.mustBeNumericArray(x);

            [n,m]  = size(x);
            obj.N      = length(x);
            obj.mX = mean(x);
            x      = x - mean(x); % center x
            % make sure x is a row vector
            if n > m
                x = transpose(x);
            end

            % check if L0 is not provided; if so, set it to N/2
            if nargin < 2
                L0 = floor(obj.N/2);
            end

            % check L0 is a positive scalar less than the number of observations
            L0 = obj.validateL0(L0);

            % 1.Embedding:
            % make the Hankel matrix of x (trajectory matrix)
            Hx = obj.embedding(x,L0);
            % 2.SVD:
            % perform the Singular Value Decomposition on Hx
            [U, S, V] = svd(Hx,'econ');
            % assign properties
            obj.L = L0;
            obj.x = x;
            obj.U = U;
            obj.S = S;
            obj.V = V;
            obj.H = Hx;
        end % end constructor

        function y = reconstruction(obj,r0)
            % RECONSTRUCTION Reconstructs the signal using a subset of singular values.
            %   y = reconstruction(obj, r0) reconstructs the signal from its trajectory matrix
            %   singular value decomposition (SVD). The reconstruction is based on selected
            %   singular values.
            %
            %   Input:
            %   - r0: Specifies which singular values to use for the reconstruction.
            %       If r0 is a scalar, the first r0 singular values are used.
            %       If r0 is a vector (e.g., [1 2 5]), the singular values at the
            %       corresponding positions are used (e.g., the first, second, and fifth).
            %
            %   Output:
            %   - y: A vector containing the recontructed signal


            % check r0 is a non-empty numeric array with max(r0) <= L
            r = obj.validateR0(r0);

            % associate the eigentriples to array r
            Sr = diag(obj.S);
            Sr = diag(Sr(r));
            y = obj.U(:,r) * Sr * transpose(obj.V(:,r));
            % do hankelization
            y = obj.hankelization(y);
            % add mean to reconstructed signal
            y = y + obj.mX;
        end % end reconstruction

        function y = grouping(obj,G,display)
            %GROUPING groups the eigentriples according to groups in G
            %   y = grouping(obj,G,plotFlag) groups eigen-triples according to G
            %   where G is an array of numbers (e.g. G = [1 1 2 2 3]).
            %   Singular values with the same number in array G are
            %   collected in the same group (e.g. if G = [1 1 2] the first
            %   two eigen-triples are summed together and the third is
            %   considered in a separate group).
            %   display is an optional argument, which when on, plots
            %   the grouped components.

            % validate G
            G = obj.validateG0(G);

            if nargin < 3
                display = 'on';
            end

            m = max(G);
            n = length(obj.U(:,1)) + length(obj.V(:,1)) - 1;
            y = zeros(m,n);
            allPos = 1:obj.L +1;
            for ii = 1:m
                tmpPos  = allPos(ii == G);
                tmpD    = transpose(diag(obj.S));
                tmpU    = obj.U(:,tmpPos) .* repmat(tmpD(tmpPos),obj.L+1,1);
                tmpY    = tmpU * transpose(obj.V(:,tmpPos));
                y(ii,:) = obj.hankelization(tmpY) + obj.mX;
            end

            if strcmp(display,'on')
                % plot components
                figure
                for ii = 1:m
                    subplot(m,1,ii)
                    plot(y(ii,:))
                    xlim([0,obj.N])
                    title(sprintf('Component %i',ii))
                    xlabel('Obs')
                    ylabel(sprintf('x_{%i}',ii))
                end % end for ii
                hold off
            end % end if strcmp
        end % end grouping

        function Rz = bootstrap(obj,r,m)
            %BOOTSTRAP bootstraps m times SSA residuals
            %   Rz = bootstrap(obj,r,m) given a time series x and the number
            %   of eigen-triples (r) used for reconstructing the signal z
            %   generates m copies of x sampling on residuals computed by the
            %   linear regression of z on x using ordinary least squares (OLS)
            %
            %   Input:
            %   r      - Number of eigentriples used for reconstructing the signal z.
            %   m      - Number of bootstrap samples to generate.
            %
            %   Output:
            %   Rz     - A matrix of size (m, length(z)), where each row is a
            %            bootstrap sample of the reconstructed signal.

            z = obj.reconstruction(r);
            zLen = length(z);
            % compute residuals using OLS
            zt = transpose(z);
            xt = transpose(obj.x+obj.mX);
            beta = [ones(zLen,1),zt]\xt;
            olsRes = transpose(xt-[ones(zLen,1),zt]*beta);
            % true bootstrapping
            R = olsRes(randi(zLen,[m,zLen]));
            Rz = R + repmat(z,m,1);
        end % end bootstrap
        
        
        function [xM, xCi, xSamp] = forecast(obj,r0,M,numSamp,display)
            %FORECAST forecasts the signal according to basic SSA
            %   xM = forecast(obj,r,M) forecasts the signal extracted from
            %   the original series x using the recursive algorithm, M
            %   times ahead.
            %
            %   Input:
            %   r0        - A scalar or array specifying which singular values to
            %               use for the signal reconstruction. If r0 is a scalar, the method
            %               uses the first r0 singular values. If r0 is an array,
            %               it uses the singular values corresponding to the
            %               indices listed in r0 (e.g., r0 = [1, 2, 5] uses the
            %               1st, 2nd, and 5th singular values).
            %   M         - The number of periods to forecast ahead.
            %   numSamp   - (Optional) The number of bootstrap samples to generate
            %               for uncertainty estimation. Default is 100.
            %   display   - (Optional) String to control the display of results.
            %               Set to 'on' to show the fan plot of the forecast.
            %               Default is 'off'.
            %
            %   Output:
            %   xM        - A vector containing the original time series data
            %               followed by the forecasted values for the next M periods.
            %   xCi       - A matrix containing the confidence intervals for the
            %               forecasted values, calculated from bootstrap samples.
            %               The intervals are determined using the 97.5th and 2.5th
            %               percentiles.
            %   xSamp     - A matrix containing forecast values derived from bootstrap
            %               samples to assess forecast uncertainty. Each row represents a
            %               different bootstrap sample forecast.

            % check r0 is a non-empty numeric array with max(r0) <= L
            r = obj.validateR0(r0);

            % check if M is a positive scalar
            if ~isscalar(M) || M <= 0
                error('M must be a positive scalar.');
            end

            if nargin < 4
                numSamp = 100;
                display = 'off';
            end

            if nargin < 5
                display = 'off';
            end

            if max(r) == obj.L
                wMsg = sprintf(strcat('when r0 == L0 there can be numerical instability\n', ...
                    'if instability occurs try max(r0) < L0\n'));
                warning(wMsg) %#ok<SPWRN>
            end
            % reconstruct signal using the U(:,1:r0) basis vectors
            P = obj.U(:,r);
            xM = obj.forecastRecursive(obj.x,P,M);
            % add mean to outputs
            xM = xM + obj.mX;

            % do bootstrapping
            if nargout > 1 || strcmp(display,'on')
                xSamp   = zeros(numSamp,length(xM));
                xR = obj.bootstrap(r,numSamp);
                xR = xR - obj.mX;
                for ii = 1:numSamp
                    tmpZ = obj.embedding(xR(ii,:),obj.L);
                    [tmpP, ~, ~] = svd(tmpZ,'econ');
                    xSamp(ii,:) = obj.forecastRecursive(xR(ii,:), ...
                        tmpP(:,r), M);
                end % end for ii
                xSamp = xSamp(:,end-M+1:end);
                xCi = prctile(xSamp,[97.5;2.5]);
                % add mean to outputs
                xCi = xCi + obj.mX;
                xSamp = xSamp + obj.mX;
            end % end if nargout > 1

            % make fanplot
            if strcmp(display,'on')
                inSamp = floor(0.1*length(obj.x));
                Dy = 1:inSamp;
                Dn = inSamp+(1:M);
                yHist = transpose([Dy; obj.x(end-inSamp+1:end)+obj.mX]);
                yFore = transpose([Dn; xSamp]);
                fanplot(yHist,yFore)
                title('Forecast with SSA basic')
            end
        end % end forecast method

        function plotSingularValues(obj,numValues,display)
            % PLOTSINGULARVALUES Plots ordered singular values and their contributions.
            %   plotSingularValues(obj) creates two plots:
            %   1. A scree plot of the first numValues singular values.
            %   2. A bar plot of the relative cumulative contribution of each singular value
            %      to the overall signal variance.
            %
            %   Inputs:
            %       numValues   - The number of singular values to plot (default is obj.L).
            %       display     - (optional) A string that specifies the type of plot:
            %                      'double' (default) for both singular values and contributions,
            %                      'cm' for only contributions,
            %                      'scree' for only singular values,
            %                      'none' for no plot.

            if nargin <2
                numValues = min(obj.L, 30);
                display = 'double';
            end

            if nargin < 3
                display = 'double';
            end

            % check numValues is a positive scalar less than or equal to L
            obj.validateNumVal(numValues);

            D = diag(obj.S);
            Drel = cumsum(D) / sum(D);

            % make plot
            figure
            switch lower(display)
                case 'double'
                    % plot singular values
                    subplot(2,1,1)
                    stem(D(1:numValues),'filled')
                    title(sprintf('First %i Singular Values',numValues))
                    xlabel('Lags')
                    ylabel('singular values')
                    % plot relative singular values
                    subplot(2,1,2)
                    bar(Drel(1:numValues))
                    xlabel('Lags')
                    ylabel('relative contribution')
                    title(sprintf('Cumulated Singular Values:\n Relative contribution to signal variance'))
                case 'cm'
                    bar(Drel(1:numValues))
                    xlabel('Lags')
                    ylabel('relative contribution')
                    title(sprintf('Cumulated Singular Values:\n Relative contribution to signal variance'))
                case 'scree'
                    stem(D(1:numValues),'filled')
                    title(sprintf('First %i Singular Values',numValues))
                    xlabel('Lags')
                    ylabel('singular values')
                otherwise
                    error(strcat('Available display options are:', ...
                        'double, scree, cm'))
            end % end switch

        end % end plotsingularvalues

        function C = wcorrelation(obj,G, display)
            %WCORRELATION returns the w-correlation matrix of two series
            %   C = wcorrelation(obj,G) returns a symmetric matrix C of
            %   weighted correlation coefficients calculated from an input
            %   nvar-by-nobs matrix Y where columns are observations and
            %   rows are variables, and an input 1-by-nobs vector w of
            %   weights for the observations.

            if nargin < 3
                display = 'on';
            end 

            % validate G
            obj.validateG0(G);

            Y = obj.grouping(G,'off');
            [~, nObs] = size(Y); % nobs: number of observations; nvar: number of variables
            % ---------------- compute weights ---------------
            w=zeros(1,nObs);
            L0 = obj.L + 1;
            w(1, 1:L0) = 1:L0;
            w( (L0+1):(nObs-L0+1) ) = L0 * ones(1, nObs-2*L0+1);
            w( (nObs-L0+2):nObs ) = nObs*ones(1,L0-1) - ( (nObs-L0+1):nObs - 1);
            % ------------------------------------------------
            wMean = (Y * w')./sum(w);             % weighted means of Y
            temp = Y - repmat(wMean, 1, nObs);    % center Y by remove weighted means
            temp = temp * transpose(temp .* w);   % weighted covariance matrix
            temp = 0.5 * (temp + temp');          % Must be exactly symmetric
            R = diag(temp);
            C = temp ./ sqrt(R * R');             % Matrix of Weighted Correlation Coefficients
            % -------------------------------------------------
            % plot w-correlation matrix
            if strcmp(display,'on')
                figure
                heatmap(abs(C));
                title('w-correlation matrix')
            end % end if
        end % end wcorrelation

        function scatterplotsEigenvectors(obj,G)
            % SCATTERPLOTSEIGENVECTORS Scatter-plots of the paired
            % singularvectors according to groups in G
            %   scatterplotseigenvectors(obj,G) makes plots of paired
            %   eingevectors in order to underline the periodicity of their
            %   corresponding component

            % validate G
            obj.validateG0(G);
            
            lenG = length(G);
            maxGroup = max(G);
            allPos = 1:lenG;
            % draw figure
            figure
            for k=1:maxGroup
                indices = allPos(G==k);
                if length(indices) == 2
                    tmpX = obj.V(:,indices);
                    subplot(maxGroup,1,k)
                    %scatter(tmpX(:,1),tmpX(:,2),".")
                    line(tmpX(:,1),tmpX(:,2))
                    grid on;
                    title(sprintf('Scatterplot of Group_{%i}',k))
                    xlabel(sprintf('V_{%i}', indices(1)));
                    ylabel(sprintf('V_{%i}', indices(2)));
                    axis equal; % Equal scaling for both axes
                else
                    fprintf('Component %i corresponds to %i singular vectors; scatter plot not possible.\n', k, length(indices));
                end % end if
            end % end k
        end %end scatterplotseigenvectors

        function [best_r0, best_rmse] = crossval_r0(obj,qInSample,numTest,display)
            % CROSSVAL_R0 does the cross-validation eigen-triples number r0
            %   best_r0 = crossval_r0(obj,qInSample,numTest) takes as optional
            %   inputs p the proportion of sample used for cross-validation
            %   (in-sample) and the number of trials (numTest) and
            %   gives the number of eigen-triples which minimizes the total
            %   rmse (in-sample + out-of-sample).
            %   [best_r0, best_rmse] = crossval_r0(obj,qInSample,numTest) provides
            %   also the root mean square error of best_r0.

            if nargin < 2
                qInSample = 0.9;
                numTest = 100;
                display = 'on';
            end

            if nargin <= 3
                numTest = 100;
                display = 'on';
            end

            % check qInSample is a number between 0 and 1
            if not(isnumeric(qInSample)) || qInSample < 0 || qInSample > 1
                error('qInSample must be a number between 0 and 1.');
            end

            % set cross-val configuration
            numInSamp = floor(length(obj.x)*qInSample);
            X0 = obj.x + obj.mX;
            inX  = X0(1:numInSamp);
            outX = X0(numInSamp+1:end);
            L0 = obj.L;
            tmpSSA = ssaBasic(inX,L0);
            [~, max_r0] = tmpSSA.checkMaxSingularValues();
            array_test = fix(linspace(2,max_r0,numTest));
            % pre-allocate output of tests
            inErr = zeros(numTest,1);
            outErr  = zeros(numTest,1);            
            for ii = 1:numTest
                % in sample error
                tmpX = tmpSSA.reconstruction(array_test(ii));
                inErr(ii) = rmse(inX,tmpX);
                % out-sample error
                tmpX = tmpSSA.forecast(array_test(ii),length(outX));
                outErr(ii) = rmse(outX,tmpX(numInSamp+1:end));
            end
            % total error (in-sample + out-sample)
            totErr = (1-qInSample)*inErr + qInSample*outErr;
            [best_rmse, best_r0] = min(totErr);
            % [best_rmse, best_r0] = min(outErr);
            best_r0 = array_test(best_r0);
            if strcmp(display,'on')
                % do the figure
                figure
                plot(array_test,log(outErr),'LineStyle','none','Marker','diamond','MarkerSize',7,'LineWidth',1)
                hold on
                plot(array_test,log(inErr),'LineStyle','none','Marker','square','MarkerSize',7,'LineWidth',1)
                plot(array_test,log(totErr),'LineStyle','-','LineWidth',1.5)
                title(sprintf('Cross-validation r with L_{optimal} = %i', obj.L))
                xlabel('r')
                ylabel('RMSE (log-scale)')
                xlim([array_test(1),array_test(end)])
                legend({'outError','inError','total'},'Location','northeastoutside')
                grid on;
                hold off;
            end % end if
        end % end crossval_r0

        function [best_L0, best_rmse] = crossval_L0(obj,r0,qInSample,numTest,display)
            % CROSSVAL_L0 does the cross-validation of number of lags L0
            %   best_L0 = crossval_r0(obj,r0,qInSample,numTest) given the number of
            %   eigen-triples r0, tests the best number of lags L0. 
            %   It takes as optional inputs qInSample, the proportion of sample
            %   for cross-validation (in-sample) and the number of 
            %   trials (numTest). best_L0 is the number of lags which 
            %   minimizes the total rmse (in-sample + out-of-sample).
            %   [best_L0, best_rmse] = crossval_L0(obj,qInSample,numTest) provides
            %   also the root mean square error of best_L0.

            % check r0 is a non-empty numeric array with max(r0) <= L
            r0 = obj.validateR0(r0);

            if nargin < 3
                qInSample = 0.9;
                numTest = 100;
                display = 'on';
            end

            if nargin <= 4
                numTest = 100;
                display = 'on';
            end

            % check qInSample is a number between 0 and 1
            if not(isnumeric(qInSample)) || qInSample < 0 || qInSample > 1
                error('qInSample must be a number between 0 and 1.');
            end

            % set cross-val configuration
            numInSamp = floor(length(obj.x)*qInSample);
            X0 = obj.x + obj.mX;
            inX  = X0(1:numInSamp);
            outX = X0(numInSamp+1:end);
            max_L0 = floor(numInSamp/2);
            min_L0 = max(r0); 
            array_test = floor(linspace(min_L0,max_L0,numTest));
            % pre-allocate output of tests
            inErr  = zeros(numTest,1);
            outErr = zeros(numTest,1);

            for ii = 1:numTest
                % set ssa model
                tmpSSA = ssaBasic(inX,array_test(ii));
                % in-sample rmse
                tmpX = tmpSSA.reconstruction(r0);
                inErr(ii) = rmse(inX,tmpX+tmpSSA.mX);
                % out-sample rmse
                tmpX = tmpSSA.forecast(r0,length(outX));
                outErr(ii) = rmse(outX,tmpX(numInSamp+1:end));
            end

            % total error (in-sample + out-sample)
            totErr = (1-qInSample)*inErr + qInSample*outErr;
            [best_rmse, best_L0] = min(totErr);
            % [best_rmse, best_L0] = min(outErr);
            best_L0 = array_test(best_L0);
            % do the figure
            if strcmp(display,'on')
                figure
                plot(array_test,log(outErr),'LineStyle','none','Marker','diamond','MarkerSize',7,'LineWidth',1)
                hold on
                plot(array_test,log(inErr),'LineStyle','none','Marker','square','MarkerSize',7,'LineWidth',1)
                plot(array_test,log(totErr),'LineStyle','-','LineWidth',1.5)
                title(sprintf('Cross-validation L with r_{prior} = %i',max(r0)))
                xlabel('L')
                xlim([array_test(1),array_test(end)])
                ylabel('RMSE (log-scale)')
                legend({'outError','inError','total'},'Location','northeastoutside')
                hold off;
                grid on;
            end % end if
        end % end crossval_L0

        function [testRMSE, xF] = backtest(obj,r0,qInSample,redNoiseMdl)
            % BACKTEST does backtesting of SSA (and if needed a redNoise model) on a signal
            %   [testRMSE, xF] = backtest(obj,r0,qInSample,redNoiseMdl) computes the
            %   Root Mean Square Error (RMSE)  of the SSA forecast on out-of-sample observations 
            %   based on qInSample and generates SSA forcast for the maximum out-of-sample period.
            %   Optionally, if redNoiseMdl is provided, backtest computes RMSE for
            %   the combination of SSA + redNoiseMdl forecast and generates the corresponding forcast
            %   for the maximum out-of-sample period.
            %
            %   Inputs:
            %       r0           - The number of eigentriples to use for forecasting with SSA.
            %       qInSample    - A vector of proportions (e.g., [0.8, 0.7]) indicating the share
            %                      of data to be used as in-sample for each backtest iteration.
            %       redNoiseMdl  - (Optional) model to fit to residuals if autocorrelation
            %                      is detected. It must have 'estimate' and 'forecast' methods implemented in MATLAB.
            %
            %   Outputs:
            %       testRMSE     - A matrix containing RMSE values for each in-sample proportion.
            %                      The first column corresponds to the RMSE of the SSA-only forecasts,
            %                      and the second column, if redNoiseMdl is provided, contains the RMSE
            %                      for combined SSA and residual noise forecast, otherwise will contain a vector of zeros.
            %       xF           - A matrix of forecasts, where the first column contains forecasts from SSA.
            %                      If a redNoiseMdl is provided, the second column includes forecasts
            %                      from the combination of SSA and the noise model. If no red noise model is
            %                      supplied, the column will contain a vector of zeros.

            % check r0 is a non-empty numeric array with max(r0) <= L
            r0 = obj.validateR0(r0);

            % check qInSample is a number between 0 and 1
            if not(isnumeric(qInSample)) || (any(qInSample < 0) || any(qInSample > 1))
                error('qInSample must be a number between 0 and 1.');
            end

            lenqInSample = length(qInSample);
            testRMSE = zeros(lenqInSample, 2);
            numObs = obj.N;
            minInSamp = floor(min(qInSample) * numObs);
            maxOutSamp = numObs - minInSamp;
            xF = zeros(maxOutSamp,2);
            useRedNoise = false;
            L0 = obj.L;

            if nargin > 3
                % Check if the redNoise model has the required 'estimate' and 'forecast' methods
                model_class = class(redNoiseMdl);
                available_methods = methods(redNoiseMdl);
                if ~ismember('estimate', available_methods) || ~ismember('forecast', available_methods)
                    error('Error: The class of the model "%s" does not have the "estimate" and "forecast" methods.', model_class);
                else
                    fprintf('The model "%s" has the "estimate" and "forecast" methods.\n', model_class);
                end
                useRedNoise = true;
            end

            for ii = 1: lenqInSample
                inSampObs = floor(qInSample(ii) * numObs);
                outSampObs = numObs - inSampObs;
                inX  = obj.x(1:inSampObs) + obj.mX;
                outX = obj.x(inSampObs+1:end);
                mySSA = ssaBasic(inX,L0);
                % SSA forecasting
                xF_SSA = mySSA.forecast(r0, outSampObs);
                xR_SSA = xF_SSA(1 : inSampObs);
                xF_SSA = xF_SSA(inSampObs + 1 : end);
                testRMSE(ii,1) = rmse(outX + obj.mX, xF_SSA);

                % forecast red noise if a model is provided
                if  useRedNoise
                    olsRes = transpose(xR_SSA - inX);
                    estMdl = estimate(redNoiseMdl,olsRes,'Display','off');
                    olsResFore = forecast(estMdl, outSampObs, olsRes);
                    xF_SSA_SARIMA = xF_SSA + transpose(olsResFore);
                    testRMSE(ii, 2) = rmse(outX + obj.mX, xF_SSA_SARIMA);
                else
                    xF_SSA_SARIMA = zeros(outSampObs,1);
                end % end if                
            end % end for ii
            xF(:,1) = xF_SSA;
            xF(:,2) = xF_SSA_SARIMA;
        end % end backtest

    end % end public methods

    methods (Access = private)

        function yNew = forecastRecursive(obj,y,P,M)
            %FORECASTRECURSIVE recursively forecasts y, M periods ahead
            %   yNew = forecastRecursive(y,P,M) applies a recursive
            %   algorithm to project y on the r-space defined by the basis
            %   vectors in P, M periods ahead.
            %
            %   Input:
            %   y      - A vector representing the time series data to be forecasted.
            %   P      - A matrix of basis vectors defining the r-space for
            %            projection.
            %   M      - The number of periods to forecast ahead.
            %
            %   Output:
            %   yNew   - A vector containing the original time series data 
            %            followed by the forecasted values for the next M periods.

            L1 = length(P(1:end-1,1));
            yLen = length(y);
            Hx = obj.embedding(y,L1);
            Xhat = P * P' * Hx; % project H on basis vectors
            Y = obj.hankelization(Xhat); % hankelization
            % apply recursion
            nu2 = sum(P(end,:).^2);
            Pup = P(1:end-1,:) .* repmat(P(end,:),L1,1);
            R   = 1/(1-nu2) * sum(Pup,2);
            yNew = zeros(1,yLen+M);
            yNew(1,1:yLen) = Y;
            for ii = 1:M
                yNew(1,yLen+ii) = yNew(1,yLen-L1+ii:yLen+ii-1) * R;
            end % end for ii
        end % end forecastRecursive

        function r = validateR0(obj,r0)
            % VALIDATER0 ensures r0 is a valid scalar or array

            if isempty(r0)
                error('ssaBasic:InvalidInput:validateR0', ...
                    'r0 must be a non-empty, positive, numeric array or scalar\n')
            end

            try
                % check r0 is positive
                mustBePositive(r0)
            catch
                
                a = class(r0);
                error('ssaBasic:InvalidInput:validateR0', strcat('r0 must be a non-empty, positive', ...
                    ', numeric array or scalar.\n Got: %s instead\n'), a);
            end

            % Check if the maximum value in r0 exceeds the allowed number of singular values
            obj.checkMaxSingularValues(r0);
            % assign an array to r
            if isscalar(r0)
                r  = 1:r0;
            else
                r = r0;
            end
        end % end validateR0

        function G = validateG0(obj,G0)
            % VALIDATEG0 ensures G0 is a valid array

            if isempty(G0) || islogical(G0)
                error('ssaBasic:InvalidInput:validateG0', 'G0 must be a positive, non-empty, numeric array\n')
            end

            try
                % check G0 is positive
                mustBePositive(G0)
            catch
                a = class(G0);
                error('ssaBasic:InvalidInput:validateG0', strcat('G0 must be a non-empty, positive', ...
                    ', numeric array or scalar.\n Got: %s instead\n'), a);
            end

            % check if the maximum value in G0 exceeds the allowed number of singular values
            obj.checkMaxSingularValues(G0);

            % check if the length of G0 exceeds the embedding dimension +1
            if length(G0) > obj.L
                error('ssaBasic:InvalidInput:validateG0','length(G0) must be less then the embedding dimension %i,',obj.L + 1)
            end

            % check for gaps in the sequence
            obj.noGaps(G0);

            % return valid G
            G = G0;
        end % end validateG0

        function numVal = validateNumVal(obj,numVal)
            %VALIDATENUMVAL ensures numVal is a valid scalar

            try
                % check if numVal is a scalar
                if ~isscalar(numVal)
                    error('ssaBasic:InvalidInput:validateNumVal', '\nnumVal must be a scalar.\n');
                end
                % make sure numVal is positive
                numVal = abs(numVal);
                % check if numValues is less than or equal to L
                if numVal > obj.L
                    error('ssaBasic:InvalidInput:validateNumVal', ...
                        'numValues = %i must be less than or equal to the embedding dimension L =  %i.\n',numVal,obj.L);
                end
            catch ME
                fprintf(ME.message);
                rethrow(ME);
            end % end try-catch
        end % end validateNumVal

        function L = validateL0(obj, L0)
            %VALIDATEL0 ensures L0 is a valid scalar

            try
                % check if L0 is a scalar
                if ~isscalar(L0)
                    error('ssaBasic:InvalidInput:validateL0', '\nL0 must be a scalar.\n'); 
                end
                % make sure L0 is positive
                L = abs(L0);
                % Check if L0 is less than half the number of observations
                N2 = floor(obj.N / 2);
                if L0 > N2
                    L = N2;
                    warning('L0 > N/2 (half of sample length). L0 set to N/2')
                end
            catch ME
                fprintf(ME.message);
                rethrow(ME);
            end % end try-catch
        end % end validateL0

        function [ft, max_singular_values] = checkMaxSingularValues(obj,r0)
            % CHECKMAXSINGULARVALUES Checks if the maximum singular values
            % do not exceed the minimum size of the trajectory matrix.
            %   [ft, max_singular_values] = checkMaxSingularValues(obj,r0)
            %   provides a logical exit code (ft == true when r0 is valid,
            %   false otherwise) and the maximum number of singular values
            %   (max_singular_values)
 
            max_singular_values = obj.L;

            if nargin < 2
                ft = true;
                return
            end

            if max(r0) > obj.L
                error('ssaBasic:InvalidInput:checkMaxSingularValues', strcat('For SSA recursive forecast, r0 must be less than L + 1 = %i. ', ...
                    'The space generated by the selected right singular vectors must not contain e_{%i}.'), ...
                    obj.L+1, obj.L+1);
            else 
                ft = true;
            end
        end % end checkMaxSingularValues

    end % end private methods

    methods (Static)

        function Hx = embedding(x,L0)
            %EMBEDDING Constructs a Hankel matrix from the input array x based on the lag L0.
            %   Hx = embedding(x,L0) takes array x as input and makes an
            %   Hankel matrix Hx of dimensions (N - L0 + 1) x (L0 + 1), where N = length(x)
            %
            %   Inputs:
            %   - x: The input array from which the Hankel matrix is constructed.
            %   - L0: The embedding dimension that determines the structure of the Hankel matrix.
            %
            %   Outputs:
            %   - Hx: The resulting Hankel matrix of size (N - L0 + 1) x L0, where
            %         each row represents overlapping segment of the input array.

            Hx = hankel(x);
            
            Hx(:,(end-L0+1):end) = []; % delete last L0 columns
            Hx = Hx(1:L0+1,:); % keep only the first L0 rows
        end % end embedding

        function y = hankelization(Y)
            %HANKELIZATION hakelization of matrix Y
            %   y = hankelization(Y) computes the averages of the
            %   anti-diagonals of matrix Y and stores the results in the
            %   array y.

            [n,m] = size(Y);
            N = n+m-1; % number of elements in the array y
            y = zeros(1,N); % a row vector
            Y = flip(Y,2); % in order to use diag
            for ii = 1:N
                kk = ii - n;
                y(ii) = mean(diag(Y,kk));
            end
            y = flip(y,2);
        end % end hankelization

        function tf = mustBeNumericArray(x)
            %MUSTBENUMERICARRAY checks x is a numeric array
            
            if not(isnumeric(x))
                a = class(x);
                error('ssaBasic:InvalidInput:mustBeNumericArray','x must be a numeric array instead is %s\n',a)
            end

            [n,m] = size(x);

            if min(n,m) > 1 || n==m
                error('ssaBasic:InvalidInput:mustBeNumericArray','x must be a numeric array')
            else
                tf = true;
            end
        end % end mustBeNumericArray

        function noGaps(G0)
            %NOGAPS check if G0 has gaps
            maxValue = max(G0);
            requiredNumbers = 1:maxValue;
            presentNumbers = unique(G0);

            if ~isequal(sort(requiredNumbers), sort(presentNumbers))
                error('ssaBasic:InvalidInput:validateG0','G0 must not have any gaps.');
            end
        end % end noGaps
    end % end of static methods
end % end classdef
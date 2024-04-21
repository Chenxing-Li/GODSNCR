function [a, k]=SparseRepresentationSolve_OMP(A, x, varargin)
%稀疏向量求解，OMP方法
%输出：稀疏向量a(N*1)，即“丰度”；实际迭代次数k
%输入：字典A(n*N)，原始像素向量x(n*1)，解向量a的长度N，最大迭代次数maxIters，
%%%%%%%迭代停止条件最后系数残余相关小于特定阈值lambdaStop，是否显示每次迭代详细过程verbose，默认不显示
%%%%%%%容差Error tolerance，默认值1e-5
% SolveOMP: Orthogonal Matching Pursuit
% Usage
%	[sols, iters, activationHist] = SolveOMP(A, x, N, maxIters, lambdaStop, solFreq, verbose, OptTol)
% Input
%	A           Either an explicit nxN matrix, with rank(A) = min(N,n) 
%               by assumption, or a string containing the name of a 
%               function implementing an implicit matrix (see below for 
%               details on the format of the function).
%	x           vector of length n.
%   N           length of solution vector. 
%	maxIters    maximum number of iterations to perform. If not
%               specified, runs to stopping condition (default)
%   lambdaStop  If specified, the algorithm stops when the last coefficient 
%               entered has residual correlation <= lambdaStop.
%   verbose     1 to print out detailed progress at each iteration, 0 for
%               no output (default)
%	OptTol      Error tolerance, default 1e-5
% Outputs
%	 sols            solution(s) of OMP
%    iters           number of iterations performed
% Description
%   SolveOMP is a greedy algorithm to estimate the solution 
%   of the sparse approximation problem
%      min ||a||_0 s.t. A*a = b
%   The implementation implicitly factors the active set matrix A(:,I)
%   using Cholesky updates. 
%   The matrix A can be either an explicit matrix, or an implicit operator
%   implemented as an m-file. If using the implicit form, the user should
%   provide the name of a function of the following format:
%     x = OperatorName(mode, m, n, a, I, dim)
%   This function gets as input a vector a and an index set I, and returns
%   x = A(:,I)*a if mode = 1, or x = A(:,I)'*a if mode = 2. 
%   A is the m by dim implicit matrix implemented by the function. I is a
%   subset of the columns of A, i.e. a subset of 1:dim of length n. a is a
%   vector of length n is mode = 1, or a vector of length m is mode = 2.
% See Also
%   SolveLasso, SolveBP, SolveStOMP
%
global isNonnegative 
isNonnegative = true;

DEBUG = 0;

STOPPING_GROUND_TRUTH = -1;
STOPPING_DUALITY_GAP = 1;
STOPPING_SPARSE_SUPPORT = 2;
STOPPING_OBJECTIVE_VALUE = 3;
STOPPING_SUBGRADIENT = 4;
STOPPING_DEFAULT = STOPPING_OBJECTIVE_VALUE;

stoppingCriterion = STOPPING_DEFAULT;

OptTol =0;
lambdaStop =1e-25;
maxIters = 1000;
[n,N]= size(A);

% Parameters for linsolve function
% Global variables for linsolve function
global opts opts_tr machPrec
opts.UT = true; 
opts_tr.UT = true; opts_tr.TRANSA = true;
machPrec = 1e-25;

% Parse the optional inputs.
if (mod(length(varargin), 2) ~= 0 ),
    error(['Extra Parameters passed to the function ''' mfilename ''' must be passed in pairs.']);
end
parameterCount = length(varargin)/2;

for parameterIndex = 1:parameterCount,
    parameterName = varargin{parameterIndex*2 - 1};
    parameterValue = varargin{parameterIndex*2};
    switch lower(parameterName)
        case 'lambda'
            lambda = parameterValue;
        case 'maxiteration'
            if parameterValue>maxIters
                if DEBUG>0
                    warning('Parameter maxIteration is larger than the possible value: Ignored.');
                end
            else
                maxIters = parameterValue;
            end
        case 'tolerance'
            OptTol = parameterValue;
        case 'stoppingcriterion'
            stoppingCriterion = parameterValue;
        case 'groundtruth'
            xG = parameterValue;
        case 'isnonnegative'
            isNonnegative = parameterValue;
        otherwise
            error(['The parameter ''' parameterName ''' is not recognized by the function ''' mfilename '''.']);
    end
end

% Initialize
a = zeros(N,1);
k = 1;
R_I = [];
activeSet = [];
res = x;
normy = norm(x);
resnorm = normy;
done = 0;

while ~done && k<maxIters
    corr = A'*res;
    if isNonnegative
        [maxcorr i] = max(corr);
    else
        [maxcorr i] = max(abs(corr));
    end
    
    if maxcorr<=0
        done = 1;
    else
        newIndex = i(1);
        
        % Update Cholesky factorization of A_I
        [R_I, done] = updateChol(R_I, n, N, A, activeSet, newIndex);
    end
    
    if ~done
        activeSet = [activeSet newIndex];
        
        % Solve for the least squares update: (A_I'*A_I)dx_I = corr_I
        dx = zeros(N,1);
        z = linsolve(R_I,corr(activeSet),opts_tr);
        dx(activeSet) = linsolve(R_I,z,opts);
        a(activeSet) = a(activeSet) + dx(activeSet);
        
        % Compute new residual
        res = x - A(:,activeSet) * a(activeSet);
        
        switch stoppingCriterion
            case STOPPING_SUBGRADIENT
                error('Subgradient is not a valid stopping criterion for OMP.');
            case STOPPING_DUALITY_GAP
                error('Duality gap is not a valid stopping criterion for OMP.');
            case STOPPING_SPARSE_SUPPORT
                error('Sparse support is not a valid stopping criterion for OMP.');
            case STOPPING_OBJECTIVE_VALUE
                resnorm = norm(res);
                if ((resnorm <= OptTol*normy) || ((lambdaStop > 0) && (maxcorr <= lambdaStop)))
                    done = 1;
                end
            case STOPPING_GROUND_TRUTH
                done = norm(xG-a)<OptTol;
            otherwise
                error('Undefined stopping criterion');
        end

        
        if DEBUG>0
            fprintf('Iteration %d: Adding variable %d\n', k, newIndex);
        end
        
        k = k+1;
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [R, flag] = updateChol(R, n, N, A, activeSet, newIndex)
% updateChol: Updates the Cholesky factor R of the matrix 
% A(:,activeSet)'*A(:,activeSet) by adding A(:,newIndex)
% If the candidate column is in the span of the existing 
% active set, R is not updated, and flag is set to 1.

global opts_tr machPrec
flag = 0;

newVec = A(:,newIndex);

if isempty(activeSet),
    R = sqrt(sum(newVec.^2));
else
    p = linsolve(R,A(:,activeSet)'*A(:,newIndex),opts_tr);

    q = sum(newVec.^2) - sum(p.^2);
    if (q <= machPrec) % Collinear vector
        flag = 1;
    else
        R = [R p; zeros(1, size(R,2)) sqrt(q)];
    end
end




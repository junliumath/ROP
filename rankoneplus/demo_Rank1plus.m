function demo_Rank1plus
% rank1plus - Rank-One Prior: Real-Time Scene Recovery
% DEMO of RANK1+;  Matlab 2019 is required.
% @ ImageType is the type of the input image
% @ 1: hazy image; 
% @ 2: sandstorm image; 
% @ 3: underwaterimage
%            
%   The Code is created based on the method described in the following paper 
%   [1] "Rank-One Prior: Toward Real-Time Scene Recovery", Jun Liu, Ryan Wen Liu, Jianing Sun, Tieyong Zeng, IEEE International Conference on Computer Vision (CVPR), 2021, 14802-14810.
%   [2] "Rank-one prior: Real-time Scene Recovery", Jun Liu, Ryan Wen Liu, Jianing Sun, Tieyong Zeng, IEEE Transaction on Pattern Analysis and Machine Intelligence, accepted, 2022.
%
%   The code and the algorithm are for non-commercial use only. 
%
%   If you are inteseted in commercial purpose, please comply with the patent: ZL 202011281893.3 and contact the authors, thank you!
% 

%  
%   Author: Jianing Sun (sunjn118@nenu.edu.cn)
%   Version : 1.0 

%%%%==================================================
 

ImageType = 3; 
param = defaultParamSetting(ImageType);

dir = 'images';
imgname = 'sandstorm1.png'; % underw1.jpg  hazy1.png sandstorm1.png
img     = im2double(imread([dir '/' imgname]));
% img = imcrop(img);

if gpuDeviceCount
    img  = gpuArray(img);    
end

 
%%%
imgvec    = reshape(img, size(img,1)*size(img,2), 3); % vectorization
%%% To update unfied spectrum 
selectedpixel = ones( size(imgvec) );
previous_basis(1,1,1:3) = [1 1 1];  
for step = 1 : 20
    % unified spectrum
    x_RGB(1 ,1, 1:3) =  mean(imgvec( selectedpixel,: ),1); 
    %  direction difference
    %  unified spectrum in each pixel
    x_mean   = repmat( x_RGB,[ size(img,1) size(img,2) 1 ] ); 
    % normalization
    spec_basis   = x_mean ./max( sqrt(sum(x_mean.^2,3)), 0.001);      
    % normalization
    imag_nmlzed  = img    ./max( sqrt(sum(img.^2,3)), 0.001);              
    % projection similarity
    proj_sim     = repmat( (((sum( spec_basis .* imag_nmlzed,3) ))),[1 1 3] ); 
    % scattering_light_estimation    
    if sum( abs(spec_basis(1,1,1:3) - previous_basis),3 ) ~= 0
        previous_basis = spec_basis(1,1,1:3);
        boots          = reshape(proj_sim, size(img,1)*size(img,2), 3);
        selectedpixel  = boots(:,1)>0.99;
    else
        break;
    end    
end

unified_spectrum = x_mean./max( sum(x_mean,3), 0.001);
% \tilde{t} is initialized.
uspec_light  = proj_sim .* sum(img,3).* unified_spectrum;


intial_img =  img;
% get_atmosphere
[ atmosphere, uspec_light ]   = get_atmosphere(  intial_img, uspec_light);
%%%  T = 1 - omega * \tilde{t}
T_ini  =  1 - param.omega * uspec_light ; 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       Rank1+:   Transmission Optimization
%
[Jr, T_tv ] = TransRefine( atmosphere, intial_img, 1-x_mean, T_ini, param );
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
%    postprocessing;  
%    For some thick-fog scenes, this operation is not recommended;
mi = prctile(Jr,param.mi,[1 2]);   
ma = prctile(Jr,param.ma,[1 2]);
Jr = ( Jr - mi)./(ma-mi);

% %
Jr = uint8( Jr*255 );
Jr = gamma0( Jr );
%


% figure(); imshow([ uint8(img*255) Jr uint8((uspec_light)*255)  uint8((1-T_tv)*255)]); 
% title('Rank1+ scenery recovery');truesize();

figure(); imshow([ uint8(img*255) Jr]); 


% save the output
imwrite(Jr, ['results/' imgname(1:end-4) '-rank1plus.png'])


 imwrite( 1-T_tv,['results/' imgname(1:end-4) '-sT-rank1plus.png'],'png');
 imwrite( 1-T_ini,['results/' imgname(1:end-4) '-T-rank1plus.png'],'png');

 
 
%          tt = imgvec;
%         tt(boots(:, 1)<=0.99,:)= 0;
%         imgselect = reshape(tt, size(img,1),size(img,2), 3);
% %         pause 
%         figure(21),  imshow([img,imgselect]),title(['step=' num2str(step)])
%         imwrite(imgselect, ['results/' imgname(1:end-4) '-selectedregion'  num2str(step)  '.png'])

end


function img = gamma0(img)
i = 0 : 255;
f = ((i + 0.5)./256 ).^(5/6);
LUT(i+1) = uint8( f.*256 -0.5 );

%%%%  rgb2hsv  - hsv2rgb            rgb2ycbcr-ycbcr2rgb
img = rgb2ycbcr(img);
img(:,:,1)    = LUT( img(:,:,1) + 1 );
img = ycbcr2rgb(img);
end

function [ atmosphere, uspec_light ] = get_atmosphere( image, uspec_light )
scatter_est = sum(uspec_light,3);
n_pixels = numel(scatter_est);
n_search_pixels = floor( n_pixels * 0.01); 

image_vec = reshape(image, n_pixels, 3);
[~, indices] = sort(scatter_est(:), 'descend');
atmosphere = mean( image_vec( indices(1:n_search_pixels), : ), 1);

atmos(1,1,:) = atmosphere;
atmosphere = repmat( atmos, [ size(scatter_est) 1 ] );

%%% To prevent over-brightness
sek = scatter_est(indices(n_search_pixels));
sek_vec = repmat( sek .* uspec_light(1,1,:)./max(scatter_est(1,1),0.001), [ size(scatter_est) 1 ]);
uspec_light = uspec_light .* repmat( scatter_est <= sek, [ 1 1 3] ) + ...
    ( 2 * sek_vec -uspec_light ) .* repmat( scatter_est > sek, [ 1 1 3] );
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%   To refine transmission map of tilde t
function [ Jr, T ] = TransRefine( atmosphere, I, sc, t0, param)

% Initializing
C = getC;
[D,Dt]  = defDDt;
MaxIter = param.iternum;
gamma   = param.gamma;
%%%%%
lambda1 = 5;     %% lambda1     50 for 0016_0.8_0.08.jpg 0001_0.8_0.2.jpg default:5;
lambda2 = 5e-1;  %% lambda2 default 5e-1;  5 for purplehaze 2,3
lambda3 = 5e-1;   %% lambda3   ???????????????????????? ?????????????????? lambda3 = 5e-1, 198.png
                         % lambda3 default 5
beta    = 10;    %% beta1 & beta2 default 10
%%%%%  Lagrange Multipliers
[ m,n,ch ] = size(t0);
Lxi1    = zeros(m,n,ch); Lxi2   = Lxi1;
Leta1   = zeros(m,n);   Leta2  = Leta1;
%%%%%%
SolRE   = 5e-4;
tao     = 1.618;

% diff
[D1I,D2I]   = D(I);
ts = mean( t0,3 );
[D1tG,D2tG]   = D(ts);
D1t = sc.* repmat( D1tG,[1 1 3] );
D2t = sc.* repmat( D2tG,[1 1 3] );

t = ts;

cont = 1;
k    = 0;
while cont
    k = k + 1;
    
    % X-subproblem
    Xterm1 = D1t - D1I + Lxi1./beta;
    Xterm2 = D2t - D2I + Lxi2./beta;
    Xterm  = sqrt(Xterm1.^2 + Xterm2.^2);
    W = exp(-gamma * (abs(D1I) + abs(D2I)));
    Xterm  = max(Xterm - (W .* lambda1)/beta, 0) ./ (Xterm + eps);
    Xmg1   = Xterm1 .* Xterm;
    Xmg2   = Xterm2 .* Xterm;
    
    % Z-subproblem
    Zterm1 = D1tG + Leta1./beta;
    Zterm2 = D2tG + Leta2./beta;
    Zterm  = sqrt(Zterm1.^2 + Zterm2.^2);
    %Zterm(Zterm == 0) = 1;
    Zterm  = max(Zterm - lambda2/beta, 0) ./ (Zterm + eps);
    Zmg1   = Zterm1 .* Zterm;
    Zmg2   = Zterm2 .* Zterm;
    %

    % t-subproblem
    zeta1X = Xmg1 + D1I - Lxi1./beta;
    zeta1Y = Xmg2 + D2I- Lxi2./beta;
    zeta2X = Zmg1 - Leta1./beta;
    zeta2Y = Zmg2 - Leta2./beta;
    %%%%%
    ttem = fft2( lambda3*ts ) + ...
       beta* sum(fft2(Dt(sc.*zeta1X, sc.*zeta1Y)),3) +...
       beta*fft2(Dt(zeta2X, zeta2Y));
    ttemp = lambda3 + beta * ( sum(sc.^2 .* C.eigsDtD,3) + C.eigsDtD2 );
    tnew = real(ifft2(ttem./(ttemp + eps)));    
    tnew(tnew <= 0) = 0;
    tnew(tnew >= 1) = 1;

    [D1tG,D2tG]   = D(tnew);
    D1t = sc.* repmat( D1tG,[1 1 3] );
    D2t = sc.* repmat( D2tG,[1 1 3] );
    
    % Updating Lagrange multipliers
    Lxi1   = Lxi1  -   tao * beta * ( Xmg1 - (D1t - D1I) );
    Lxi2   = Lxi2  -   tao * beta * ( Xmg2 - (D2t - D2I) );
    Leta1  = Leta1 -   tao * beta * ( Zmg1 - D1tG );
    Leta2  = Leta2 -   tao * beta * ( Zmg2 - D2tG );
    
    %
    re = norm(tnew(:) - t(:),'fro') / norm(t(:),'fro');
    t  = tnew;
    cont  = (k < MaxIter) && (re > SolRE);
    %
end

figure,imshow(t,[])
T = sc.*repmat(t, [1 1 3]);
Jr = ( I - atmosphere )./max(T,0.01) + atmosphere;





% % Nested function
    function C = getC
        sizeF     = size(t0);
        % psf2otf: computes the Fast Fourier Transform (FFT) of the point-spread function (PSF)
        C.eigsD1  = psf2otf([1,-1], sizeF);   
        C.eigsD2  = psf2otf([1;-1], sizeF);  
        C.eigsDtD = abs(C.eigsD1).^2 + abs(C.eigsD2).^2;  
        
        C.eigsD21  = psf2otf([1,-1], [sizeF(1) sizeF(2)]);  
        C.eigsD22  = psf2otf([1;-1], [sizeF(1) sizeF(2)]); 
        C.eigsDtD2 =  abs(C.eigsD21).^2 + abs(C.eigsD22).^2;
    end
    function [D,Dt] = defDDt
        % defines finite difference operator D 
        % and its transpose operator           
        % referring to FTVD code
        D = @(U) ForwardD(U);
        Dt = @(X,Y) Dive(X,Y);
    end
    function [Dux,Duy] = ForwardD(U) %diff     
        %%% Forward finite difference operator
        Dux = [diff(U,1,2), U(:,1,:) - U(:,end,:)];
        Duy = [diff(U,1,1); U(1,:,:) - U(end,:,:)];
    end
    function DtXY = Dive(X,Y) %Dt=-div      
        % Transpose of the forward finite difference operator
        DtXY = [X(:,end,:) - X(:, 1,:), -diff(X,1,2)];
        DtXY = DtXY + [Y(end,:,:) - Y(1, :,:); -diff(Y,1,1)];
    end
end

function param = defaultParamSetting(ImageType)
% Set the necessary parameters
% Note that the default paramameters do not mean the best parameter
% setting. If you are not satisfied with the recovered result, try to ajust the 
% parameters accordingly.
            if ImageType == 1
                param.omega = 0.99;
                param.mi    = 1;
                param.ma    = 95;
                param.gamma = 5e1;
            elseif ImageType == 2
                %%%% Although the sandstorm image looks dim, 
                %%%% it is bright enough.
                %%%% The consistency of the spectrum is good                
                param.omega = 0.8;
                param.mi    = 1;
                param.ma    = 95;
                param.gamma = 3e1;
            elseif ImageType == 3
                %%%% The underwater image is very dark, 
                %%%% but the consistency of the spectrum is good.
                param.omega = 0.75;
                param.mi    = 1;
                param.ma    = 95;
                param.gamma = 5e1;
            elseif ImageType == 4
                param.omega = 0.7;
                param.mi    = 2;
                param.ma    = 99.6;
                param.gamma = 1e2;
            end
            param.iternum = 10;
end




   

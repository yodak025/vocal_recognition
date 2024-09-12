[ya,Fs]=audioread('a_p.wav');
[ye,Fs]=audioread('e_p.wav');
[yi,Fs]=audioread('i_p.wav');
[yo,Fs]=audioread('o_p.wav');
[yu,Fs]=audioread('u_p.wav');
ya=ya-mean(ya);
ye=ye-mean(ye);
yi=yi-mean(yi);
yo=yo-mean(yo);
yu=yu-mean(yu);
N=160; % nro. de muestras por frame
D=N*Fs;  % duracion de un frame
NFa=floor(length(ya)/N);  % nro. de frames
NFe=floor(length(ye)/N);
NFi=floor(length(yi)/N);
NFo=floor(length(yo)/N);
NFu=floor(length(yu)/N);
ya=ya(1:N*NFa);    % 'y' con numero entero de frames
ye=ye(1:N*NFe);
yi=yi(1:N*NFi);
yo=yo(1:N*NFo);
yu=yu(1:N*NFu);
yframesa=reshape(ya,[N,NFa]);  % matriz con frames en cada columna
yframese=reshape(ye,[N,NFe]);
yframesi=reshape(yi,[N,NFi]);
yframeso=reshape(yo,[N,NFo]);
yframesu=reshape(yu,[N,NFu]);
yframesaw=yframesa.*kron(ones(1,NFa),hamming(N));  % frames enventanados
yframesew=yframese.*kron(ones(1,NFe),hamming(N));
yframesiw=yframesi.*kron(ones(1,NFi),hamming(N));
yframesow=yframeso.*kron(ones(1,NFo),hamming(N));
yframesuw=yframesu.*kron(ones(1,NFu),hamming(N));
ycepstruma=rceps(yframesaw);  % matriz con cepstrum en cada columna
ycepstrume=rceps(yframesew);
ycepstrumi=rceps(yframesiw);
ycepstrumo=rceps(yframesow);
ycepstrumu=rceps(yframesuw);
figure(1)
subplot(311)
plot(ycepstruma(1:20,:))
ylabel('/a/')
subplot(312)
plot(ycepstrume(1:20,:))
ylabel('/e/')
subplot(313)
plot(ycepstrumi(1:20,:))
ylabel('/i/')
figure(2)
subplot(211)
plot(ycepstrumo(1:20,:))
ylabel('/o/')
subplot(212)
plot(ycepstrumu(1:20,:))
ylabel('/u/')
patron_a=sum(ycepstruma(1:20,:),2)/NFa;
patron_e=sum(ycepstrume(1:20,:),2)/NFe;
patron_i=sum(ycepstrumi(1:20,:),2)/NFi;
patron_o=sum(ycepstrumo(1:20,:),2)/NFo;
patron_u=sum(ycepstrumu(1:20,:),2)/NFu;
figure(3)
subplot(311)
plot(patron_a)
ylabel('patron /a/')
subplot(312)
plot(patron_e)
ylabel('patron /e/')
subplot(313)
plot(patron_i)
ylabel('patron /i/')
figure(4)
subplot(211)
plot(patron_o)
ylabel('patron /o/')
subplot(212)
plot(patron_u)
ylabel('patron /u/')
cov_a=cov(ycepstruma(1:20,:)');
cov_e=cov(ycepstrume(1:20,:)');
cov_i=cov(ycepstrumi(1:20,:)');
cov_o=cov(ycepstrumo(1:20,:)');
cov_u=cov(ycepstrumu(1:20,:)');



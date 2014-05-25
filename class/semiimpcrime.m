function [A0, t, B, p] = semiimpcrime(n,tmax,dt,A0,B0),
if nargin<1, n=10; end;
if nargin<2, tmax=1; end;
if nargin<3, dt=.001; end;

[k,w,nu,D,gamma,A02,B02,p0] = parameters(n);
if nargin<4, A0=A02; end;
if nargin<5, B0=B02; end;
B(:,:,1)=B0; p(:,:,1)=p0; 

% in Short they use a 128x128 physical grid and 512x512 numerical grid
x = linspace(0,128,n+1)'; x=x(1:end-1); y=x; dx = x(2)-x(1);

t=0; ta(1)=0; i=1; imax=ceil(tmax/dt); j=1; jmax=100; iskip = floor(imax/jmax);
for j1=1:n, for j2=1:n, k1=j2-n/2-1;k2=j1-n/2-1; K(j1,j2)=k1^2+k2^2; end; end;
K=-(2*pi/dx/n)^2*fftshift(K);

lhs1 = (1+w*dt) - nu*D*dt*K;
lhs2 = 1 - dt*D*K;

Aold = A0; Bold = B0; pold = p0;
while t<tmax,
	rhs1 = Bold + dt*k*pold.*Aold;
	Bnew = real(ifft2(fft2(rhs1)./lhs1));
	Anew = A0+Bnew; 

	logA=log(Anew);
	rhsp = RHS_p(logA, pold);
    LlogA = real(ifft2(K.*fft2(logA)));
    rhs2 = pold - 2*dt*D*LlogA.*pold -dt*Anew.*pold;
	rhs2 = rhs2 - 2*D*dt*rhsp + dt*gamma;
	pnew = real(ifft2(fft2(rhs2)./lhs2));

	if mod(i+1,iskip)==1,
		B(:,:,j+1) = Bnew;
		if nargout==4, p(:,:,j+1) = pnew; end
		ta(j+1) = t+dt;
		j=j+1;
	end
	t=t+dt; i=i+1;
	Aold=Anew; Bold=Bnew; pold=pnew;
end
t=ta;

function rhs = RHS_p(logA, p),
	[px, py] = grad2(p);
	[Ax, Ay] = grad2(logA);
	rhs = px.*Ax + py.*Ay; 
end

% compute gradient w periodic BCs
function [Dx, Dy] = grad2(M),
    kn=2.*pi/n/dx;
    ik=1i*kn*[0:n/2 -(n/2)+1:-1]'; IK=repmat(ik,1,n);
    Dx=real(ifft(IK.*fft(M')))';
    Dy=real(ifft(IK.*fft(M)));
end

end

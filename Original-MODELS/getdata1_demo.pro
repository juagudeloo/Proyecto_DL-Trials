ptk='./'
nmod='154000'

npt=480

npty=256

close,1
coef=sqrt(4*3.14159265)

openr,1,ptk+'iout.'+nmod
a=assoc(1,fltarr(npt,npt))
iout=a(0)
close,1
print,iout[0,0]

openr,1,ptk+'eos.'+nmod
a=assoc(1,fltarr(npt,npty,npt))
tem=a(0)
pre=a(1)
close,1
print,tem(0,0,0)

openr,1,ptk+'result_0.'+nmod
a=assoc(1,fltarr(npt,npty,npt))
rho=a(0)
close,1

openr,1,ptk+'result_1.'+nmod
a=assoc(1,fltarr(npt,npty,npt))
vxx=a(0)
close,1

openr,1,ptk+'result_2.'+nmod
a=assoc(1,fltarr(npt,npty,npt))
vyy=a(0)
close,1

openr,1,ptk+'result_3.'+nmod
a=assoc(1,fltarr(npt,npty,npt))
vzz=a(0)
close,1

openr,1,ptk+'result_4.'+nmod
a=assoc(1,fltarr(npt,npty,npt))
ene=a(0)
close,1

openr,1,ptk+'result_5.'+nmod
a=assoc(1,fltarr(npt,npty,npt))
bxx=a(0)
close,1

openr,1,ptk+'result_6.'+nmod
a=assoc(1,fltarr(npt,npty,npt))
byy=a(0)
close,1

openr,1,ptk+'result_7.'+nmod
a=assoc(1,fltarr(npt,npty,npt))
bzz=a(0)
close,1



vxx = vxx / rho  ; convert momenta to physical velocity in cgs
vyy = vyy / rho
vzz = vzz / rho


bxx = bxx * coef ; conversion to Gauss
byy = byy * coef
bzz = bzz * coef







end

def read(self,verbose=1):

    sizerec = self.npix

    print "reading EOS"
    self.mtpr = n.fromfile(self.ptm+"eos."+self.filename,dtype=n.float32)
    self.mtpr = self.mtpr.reshape((2,self.nx,self.nz,self.ny),order="C")
    print "EOS done")

    print(self.mtpr[0,0,0,0],self.mtpr[1,0,0,0])

    print "reading rho"
    self.mrho = n.fromfile(self.ptm+"result_0."+self.filename,dtype=n.float32)
    self.mrho = self.mrho.reshape((self.nx,self.nz,self.ny),order="C")
    print "rho done"

    print "reading vxx"
    self.mvxx = n.fromfile(self.ptm+"result_1."+self.filename,dtype=n.float32)
    self.mvxx = self.mvxx.reshape((self.nx,self.nz,self.ny),order="C")
    print "vxx done"

    print "reading vyy"
    self.mvyy = n.fromfile(self.ptm+"result_2."+self.filename,dtype=n.float32)
    self.mvyy = self.mvyy.reshape((self.nx,self.nz,self.ny),order="C")
    print "vyy done"

    print "reading vzz"
    self.mvzz = n.fromfile(self.ptm+"result_3."+self.filename,dtype=n.float32)
    self.mvzz = self.mvzz.reshape((self.nx,self.nz,self.ny),order="C")
    print "vzz done"

    print "reading bxx"
    self.mbxx = n.fromfile(self.ptm+"result_5."+self.filename,dtype=n.float32)
    self.mbxx = self.mbxx.reshape((self.nx,self.nz,self.ny),order="C")
    print "bxx done"

    print "reading byy"
    self.mbyy = n.fromfile(self.ptm+"result_6."+self.filename,dtype=n.float32)
    self.mbyy = self.mbyy.reshape((self.nx,self.nz,self.ny),order="C")
    print("byy done"

    print "reading bzz"
    self.mbzz = n.fromfile(self.ptm+"result_7."+self.filename,dtype=n.float32)
    self.mbzz = self.mbzz.reshape((self.nx,self.nz,self.ny),order="C")
    print "bzz done"


    self.mvxx=self.mvxx/self.mrho
    self.mvyy=self.mvyy/self.mrho
    self.mvzz=self.mvzz/self.mrho
    coef=math.sqrt(4.0*3.14159265)
    self.mbxx=self.mbxx*coef
    self.mbyy=self.mbyy*coef
    self.mbzz=self.mbzz*coef

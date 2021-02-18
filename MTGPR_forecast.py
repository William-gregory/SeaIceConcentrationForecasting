import operator
import random
import glob
import numpy as np
from numpy.linalg import multi_dot as mdot
import math
import os
from scipy import stats
from scipy.spatial.distance import squareform,pdist,cdist
import scipy.optimize
import pickle
import datetime
import itertools
import CN as Network

def CNs(month,areas,ymax):
    Net = {}
    data = np.zeros(SIC[str(month)+'_dt_'+str(ymax)].shape)*np.nan
    data[sector] = SIC[str(month)+'_dt_'+str(ymax)][sector]
    print('Creating network: 1979 - ',ymax)
    net = Network(dimX=dimX,dimY=dimY)
    Network.cell_level(net, data, "25sqkm_ESSLap_79-"+str(ymax), datapath)
    Network.tau(net, data, 0.01, "25sqkm_ESSLap_79-"+str(ymax), datapath)
    Network.area_level(net, data)
    Network.intra_links(net, data, area=areas)
    Net[str(month)+'_nodes_'+str(ymax)] = net.V
    Net[str(month)+'_anoms_'+str(ymax)] = net.anomaly
    return Net

def nearPSD(K,epsilon=1e-12):
    """
    return the nearest positive semi-definite matrix to K
    """
    Lambda, U = np.linalg.eigh(K)
    Lambda = np.maximum(Lambda,epsilon)
    K_ = np.dot(np.sqrt(np.diag(1/(np.dot(U*U,Lambda)))),U) * np.sqrt(Lambda)
    return np.dot(K_,K_.T)

def SGPkernel(x,xs=None,grad=False,ell=1):
    if xs is None:
        Q = squareform(pdist(np.sqrt(3.)*x/ell,'euclidean'))
        k = (1 + Q) * np.exp(-Q)
        dk = np.zeros((len(ell),k.shape[0],k.shape[1]))
        for theta in range(len(ell)):
            q = squareform(pdist(np.sqrt(3.)*np.atleast_2d(x[:,theta]/ell[theta]).T,'euclidean'))
            dk[theta,:,:] = q * q * np.exp(-Q)
    else:
        Q = cdist(np.sqrt(3.)*x/ell,np.sqrt(3.)*xs/ell,'euclidean')
        k = (1 + Q) * np.exp(-Q)
    if grad:
        return k,dk
    else:
        return k
    
def SMLII(hypers,x,y):
    ell = [np.exp(hypers[0]),np.exp(hypers[1])]
    sn2 = np.exp(hypers[2])
    n = len(y)
    Kx,dK = SGPkernel(x,grad=True,ell=ell)
    try:
        L = np.linalg.cholesky(Kx + np.eye(n)*sn2)
        A = np.atleast_2d(np.linalg.solve(L.T,np.linalg.solve(L,y))).T
        nlZ = np.dot(y.T,A)/2 + np.log(L.diagonal()).sum() + n*np.log(2*np.pi)/2

        Q = np.linalg.solve(L.T,np.linalg.solve(L,np.eye(n))) - np.dot(A,A.T)
        dnlZ = np.zeros(len(hypers))
        for theta in range(len(hypers)):
            if theta < 2:
                dnlZ[theta] = (Q*dK[theta,:,:]).sum()/2
            else:
                dnlZ[theta] = sn2*np.trace(Q)
    except np.linalg.LinAlgError as e:
        nlZ = np.inf ; dnlZ = np.ones(len(hypers))*np.inf
    return nlZ,dnlZ

def lengthscales(dX,dY,ymin,ymax):
    n = ymax-1 - ymin + 1
    res = {}
    lx_yrs = np.zeros((dX,dY,n))*np.nan
    ly_yrs = np.zeros((dX,dY,n))*np.nan
    sn2_yrs = np.zeros((dX,dY,n))*np.nan
    for yr in range(ymin,ymax):
        conc = SIC['sep_dt_'+str(ymax-1)][:,:,yr-1979][sector]
        X = np.array([SIC['x'][sector],SIC['y'][sector]]).T
        X_tree = scipy.spatial.cKDTree(X)
        N = X.shape[0]
        l = np.zeros((N,4))*np.nan ; sn2 = np.zeros((N,4))*np.nan
        for index in range(N):
            idr = X_tree.query_ball_point(x=X[index,:], r=375e3)
            halves = [np.where(X[idr,0]>X[index,0]),np.where(X[idr,0]<X[index,0]),\
                  np.where(X[idr,1]>X[index,1]),np.where(X[idr,1]<X[index,1])]
            q = 0
            for half in halves:
                X_half = X[idr][half]
                z_half = conc[idr][half]
                d = np.sqrt((X[index,0]-X_half[:,0])**2 +\
                                           (X[index,1]-X_half[:,1])**2)
                R = 1 - (conc[index]-z_half)**2/(2*np.var(z_half))
                R[R<0] = 0
                lsq = lambda ell : np.sum((R - (1+d/np.exp(ell))*np.exp(-d/np.exp(ell)))**2) #Ricker
                l[index,q] = np.exp(scipy.optimize.minimize(lsq,x0=np.log(25e3),jac=False).x)
                sn2[index,q] = np.var(z_half)
                q += 1
        lx_yrs[:,:,yr-ymin][sector] = np.nanmean([l[:,0],l[:,1]],0)
        ly_yrs[:,:,yr-ymin][sector] = np.nanmean([l[:,2],l[:,3]],0)
        sn2_yrs[:,:,yr-ymin][sector] = np.nanmean(sn2,1)
    res['ell_x_yrs'] = lx_yrs
    res['ell_y_yrs'] = ly_yrs
    res['sn2_yrs'] = sn2_yrs
    res['ell_x'] = np.nanmean(lx_yrs,2)
    res['ell_y'] = np.nanmean(ly_yrs,2)
    res['sn2'] = np.nanmean(sn2_yrs,2)
    
    with open(datapath+'/Kf_hypers_25km_ESS_Laptev_1979-'+str(ymax-1)+'_average_375kmiter_Matern.pkl','wb') as f:
        pickle.dump(res,f)

def MTGPR(dX, dY, year):
    """
    Multi-task Gaussian Process Regression using a non-stationary inter-task covariance matrix, see equations (5,6)
    of Paciorek and Schervish, 2005. Inference and prediction is done based on the efficient approach of Rakitsch
    et al, 2013. 
    """
    print('Forecast year = ',year)
    fs = np.zeros((dX,dY))
    sfs2 = np.zeros((dX,dY))
    fs_r = np.zeros((dX,dY))
    
    def MLII(hypers):
        """
        negative log marginal likelihood objective function, used as part of numerical optimisation to determine
        remaining hyperparameters.
        """
        sf2_x = np.exp(hypers[0])
        ell_x = np.exp(hypers[1])
        sf2_y = np.exp(hypers[2])
        print(np.exp(hypers))
        try:
            Sigma = scipy.linalg.expm(ell_x*M)
            Kx = np.linalg.multi_dot([X,Sigma,X.T])
            Kf_ = mdot([np.diag(LD**-.5),UD.T,sf2_y*Kf,UD,np.diag(LD**-.5)])
            Kx_ = mdot([np.diag(LI**-.5),UI.T,sf2_x*Kx,UI,np.diag(LI**-.5)])
            Lf_, Uf_ = np.linalg.eigh(Kf_)
            Lx_, Ux_ = np.linalg.eigh(Kx_)
            Lff_,Lxx_ = np.meshgrid(Lf_,Lx_)
            Li = np.diag(1/(np.kron(Lf_,Lx_) + np.ones(n*T)))
            V = mdot([Ux_.T,np.diag(LI**-.5),UI.T,y,UD,np.diag(LD**-.5),Uf_]).ravel(order='F')

            nlZ = .5*n*np.log(LD).sum() + .5*T*np.log(LI).sum() + .5*np.log(Lff_*Lxx_ + 1).sum()\
                  + mdot([V,Li,V.T])/2 + n*T*np.log(2*np.pi)/2
        
            dnlZ = np.zeros(len(hypers))
            for theta in range(len(hypers)):
                if theta == 0:
                    dK = 2 * sf2_x * Kx
                    dnlZ[theta] = .5*np.dot(Li.diagonal(),np.kron(Lf_,mdot([Ux_.T,np.dot(dK,Kx_),Ux_]).diagonal())) -\
                                       .5*np.dot(np.dot(Li,V),mdot([Ux_.T,np.dot(dK,Kx_),Ux_,np.dot(Li,V).reshape(n,T,order='F'),np.diag(Lf_)]).ravel(order='F'))
                elif theta == 1:
                    dK = sf2_x * np.linalg.multi_dot([X,np.dot(M,Sigma),X.T])
                    dnlZ[theta] = .5*np.dot(Li.diagonal(),np.kron(Lf_,mdot([Ux_.T,np.dot(dK,Kx_),Ux_]).diagonal())) -\
                                       .5*np.dot(np.dot(Li,V),mdot([Ux_.T,np.dot(dK,Kx_),Ux_,np.dot(Li,V).reshape(n,T,order='F'),np.diag(Lf_)]).ravel(order='F'))
                elif theta == 2:
                    dK = 2 * sf2_y * Kf
                    dnlZ[theta] = .5*np.dot(Li.diagonal(),np.kron(Lx_,mdot([Uf_.T,np.dot(dK,Kf_),Uf_]).diagonal())) -\
                                       .5*np.dot(np.dot(Li,V),mdot([Uf_.T,np.dot(dK,Kf_),Uf_,np.dot(Li,V).reshape(n,T,order='F').T,np.diag(Lx_)]).ravel(order='F'))
        except (ValueError, np.linalg.LinAlgError) as e:
            nlZ = np.inf ; dnlZ = np.ones(len(hypers))*np.inf
        return nlZ,dnlZ
    
    y = SIC['sep_dt_'+str(year-1)][sector].T
    n,T = y.shape
    print('number of years:',n)
    print('number of tasks:',T)

    if os.path.exists(datapath+'/Kf_hypers_25km_ESS_Laptev_1979-'+str(year-1)+'_average_375kmiter_Matern.pkl'):
        with open(datapath+'/Kf_hypers_25km_ESS_Laptev_1979-'+str(year-1)+'_average_375kmiter_Matern.pkl','rb') as f:
            thetas = pickle.load(f)
    else:
        lengthscales(dX,dY,1979,year)
        with open(datapath+'/Kf_hypers_25km_ESS_Laptev_1979-'+str(year-1)+'_average_375kmiter_Matern.pkl','rb') as f:
            thetas = pickle.load(f)
    
    sn2 = thetas['sn2'][sector]
    
    X = []
    for area in Net['aug_nodes_'+str(year)]:
        X.append(Net['aug_anoms_'+str(year)][area])
    X = np.asarray(X).T

    Xs = np.asarray([X[-1,:]]) #test inputs
    X = X[:-1,:] #training inputs
    N = X.shape[1]
    print('number of predictors:',N)
    
    M = np.abs(np.cov(X,rowvar=False,bias=True))
    np.fill_diagonal(M,0)
    np.fill_diagonal(M,-M.sum(0))
    
    ell = np.array([thetas['ell_x'][sector],thetas['ell_y'][sector]]).T
    Sigmax = np.prod(ell,axis=1)**.25
    x = np.array([SIC['x'][sector],SIC['y'][sector]]).T
    ellx1,ellx2 = np.meshgrid(ell[:,0],ell[:,0])
    elly1,elly2 = np.meshgrid(ell[:,1],ell[:,1])
    ell_ = np.array([(ellx1+ellx2)/2,(elly1+elly2)/2]).transpose(1,2,0)
    Sigmai,Sigmaj = np.meshgrid(Sigmax,Sigmax)
    Sigmaij = np.prod(ell_,2)**-.5

    x1,x2 = np.meshgrid(x[:,0],x[:,0])
    y1,y2 = np.meshgrid(x[:,1],x[:,1])
    
    xi = np.sqrt(3.)*np.reshape(np.array([x1,y1]).transpose(1,2,0)/ell_,(x.shape[0]*x.shape[0],2))
    xj = np.sqrt(3.)*np.reshape(np.array([x2,y2]).transpose(1,2,0)/ell_,(x.shape[0]*x.shape[0],2))
    Qij = np.reshape(np.sqrt((xi[:,0]-xj[:,0])**2 + (xi[:,1]-xj[:,1])**2),(x.shape[0],x.shape[0]))
    Kf = Sigmai * Sigmaj * Sigmaij * (1 + Qij) * np.exp(-Qij)

    D = np.diag(sn2)
    LI = np.ones(n)
    UI = np.diag(LI)
    LD, UD = np.linalg.eigh(D)
    
    #hypers = scipy.optimize.minimize(MLII,x0=[np.log(5e-6),np.log(5e-6),np.log(1e6)],jac=True,method='CG').x
    #print('hypers: ',np.exp(hypers))
    #sf2_x = np.exp(hypers[0]) ; ell_x = np.exp(hypers[1]) ; sf2_y = np.exp(hypers[2])
    sf2_x = 5e-6 ; ell_x = 5e-6 ; sf2_y = 1e5

    Kf = sf2_y * Kf
    Kf_ = mdot([np.diag(LD**-.5),UD.T,Kf,UD,np.diag(LD**-.5)])
    Lf_, Uf_ = np.linalg.eigh(Kf_)

    Sigma = scipy.linalg.expm(ell_x*M)
    Kx = sf2_x * np.linalg.multi_dot([X,Sigma,X.T])
    Kxsx = sf2_x * np.linalg.multi_dot([Xs,Sigma,X.T])
    Kxs = sf2_x * np.linalg.multi_dot([Xs,Sigma,Xs.T])
            
    Kx_ = mdot([np.diag(LI**-.5),UI.T,Kx,UI,np.diag(LI**-.5)])
    Lx_, Ux_ = np.linalg.eigh(Kx_)
    Li = np.diag(1/(np.kron(Lf_,Lx_) + np.ones(n*T)))
            
    V = mdot([Ux_.T,np.diag(LI**-.5),UI.T,y,UD,np.diag(LD**-.5),Uf_]).ravel(order='F')
    y_ = np.dot(Li,V).reshape(n,T,order='F')
    
    fs[sector] = mdot([Kxsx,UI,np.diag(LI**-.5),Ux_,y_,Uf_.T,np.diag(LD**-.5),UD.T,Kf.T])
    alpha = np.kron(np.dot(Kf,mdot([Uf_.T,np.diag(LD**-.5),UD.T]).T),np.dot(Kxsx,Ux_))
    sfs2[sector] = (np.kron(Kf,Kxs) - mdot([alpha,Li,alpha.T])).diagonal() + sn2
    trendline = (np.asarray([np.arange(n+1)]*T).T*SIC['sep_trend_'+str(year-1)][sector][:,0]) + SIC['sep_trend_'+str(year-1)][sector][:,1]
    fs_r[sector] = np.maximum(np.minimum(fs[sector] + trendline[-1,:],1),0)
    
    GP['fmean_'+str(year)] = fs
    GP['fvar_'+str(year)] = sfs2
    GP['fmean_rt_'+str(year)] = fs_r

def save(dic,path):
    with open(path,'wb') as f:
        pickle.dump(dic,f)

datapath = '~/SIC_forecast'

with open(datapath+'/SIC_55N_all_months_25x25_py2p7.pkl','rb') as f:
    SIC = pickle.load(f)

ymin = 2007
ymax = 2007
    
SIC['x'] = np.load(datapath+'/x_ESS_laptev_25x25.npy')
SIC['y'] = np.load(datapath+'/y_ESS_laptev_25x25.npy')
dimX = 448
dimY = 304

mask = np.load(datapath+'/NSIDC_regions.npy')
ESS = np.where(mask==11)
Laptev = np.where(mask==10)
sector = (np.concatenate((ESS[0],Laptev[0])),np.concatenate((ESS[1],Laptev[1])))

if os.path.exists(datapath+'/ESSLap_networks_aug_'+str(ymax)+'.pkl'):
    with open(datapath+'/ESSLap_networks_aug_'+str(ymax)+'.pkl','rb') as f:
        Net = pickle.load(f)
else:
    Net = CNs('aug',SIC['psa'],ymax)
    save(Net,datapath+'/ESSLap_networks_aug_'+str(ymax)+'.pkl')

GP = {}
for year in range(ymin,ymax+1):
    MTGPR(dimX,dimY,year)

save(GP,datapath+'/MTGPR_forecast_ESS_Lap_'+str(ymin)+'-'+str(ymax)+'.pkl')
#!/usr/bin/env python
import numpy as np
import math
import random
import time

# This was created to separate make_data and the model and the solver
rnd = np.random
#rnd.seed(0)
#This version is changed according the JMetal 
def make_data(U1,V1,K1,N1,Psi_u1,Psi_u2,Phi_u1,Phi_u2,B_u1,B_u2,r_u1,tau_v1,tau_v2,sigma_v1,sigma_v2,b_v1,b_v2,L_v1,R_v1):

	
	#Slice1N( U1,  V1,  K1, N1,  dmaxv1, dminv1,  Bmaxu1!,  Bminu1!,  bmaxuv1,  bminuv1,  UpSpi1,  UpPhi1,  upSpimax1,  upSpimin1,  upPhimax1,  upPhimin1,  uptaumax1,  uptaumin1,  upsigmamax1,  upsigmamin1,  Lv1,  ru1,  ru2,ru3,rv1,rv2,rv3, q1,I1)));
    mec = U1 #U number of MECs
    U,V,E,Psi_u,Phi_u,B_u,r_u,tau_v,sigma_v,b_v,L_v,R_v= {},{},{},{},{},{},{},{},{},{},{},{},{},{}
    #xmec = rnd.rand(mec)*100
    #ymec = rnd.rand(mec)*100
    dp = V1 #V number of demand points
    
    U = [u for u in range(mec)] # the set of MECs
    V = [v for v in range(dp)] # the set of demand points
    C_u =[]
    
    
    #xdp = rnd.rand(dp)*100
    #ydp = rnd.rand(dp)*100
	
	
    K = K1 #number of redundancy
    N = N1 #N Number of demand points sharing a backup slice
	
	
    #PMEC = {u: (xmec[u],ymec[u]) for u in range(U1) }
    #PDP  = {v: (xdp[v],ydp[v]) for v in range(V1)}
	
    # Psi_u MEC CPU capacity in MIPS
    Psi_u = [rnd.randint(Psi_u1, Psi_u2) for u in range(U1)]
    # Phi_u MEC RAM capacity in GB 6 GB -> 48000 Megabit
    Phi_u = [rnd.randint(Phi_u1, Phi_u2) for u in range(U1)]
    #MIPSfactor = PCI / CP / 3600
    #CP: the number of processors: 25 for a 2964-725.
    #PCI: Processor Capacity Index.
    
    #B_u MEC total bandwidth capacity in Mbps
    B_u = [rnd.randint(B_u1, B_u2) for u in range(U1)]
	
    
    #b_v Bandwidth consumed by the demand point v in Mbps
    
    #b_v = [ random.randint(b_v1, b_v2) for v in range(V1)]
    
    #C = {(i,j):np.hypot(xc[i]-xc[j],yc[i]-yc[j])for i,j in E} # Euclidean Distance
    #b_v = {(i,j):random.randint(b_v1, b_v2) for i,j in E} 
    #L_v  Bound on the latency requirement of the demand point v in ms
    L_v = [ L_v1 for v in range(V1)]
    
    R_v = [ R_v1 for v in range(V1)]
    
    #r_u Failure probability of the MEC u \in U
    r_u = [r_u1 for u in range(U1)] 
    
    #tau_v Processing demand of the demand point v in MIPS
    tau_v = [rnd.randint(tau_v1, tau_v2) for v in range(V1)]
    #sigma_v Memory demand of the demand point v in GB = x8000 MBit
    sigma_v = [rnd.randint(sigma_v1, sigma_v2) for v in range(V1)]
    #C_u Maximum possible number of slices in the MEC u
    C_u1 = {u: (Psi_u[u])/min(tau_v[v] for v in range(V1) ) for u in range(U1)  }
    C_u2 = {u: (Phi_u[u])/min(sigma_v[v] for v in range(V1) ) for u in range(U1)  }
    C_u = [int(math.ceil(min(C_u1[u],C_u2[u]))) for u in range(U1)]
    #C_ui = {(u,i) for u in range(U1) for i in range (C_u[u])  }
    #C_uik = {(u,i,k) for u in range(U1) for i in range (C_u[u]) for k in range(K)  }
    E = {(u,v,i,k) for u in U for v in V for i in range(C_u[u])for k in range(K)} # The set of edges
    b = {}	
	
    for v in V:
        b[v] = rnd.randint(b_v1, b_v2)
    b_v = {(u,v,i,k):b[v] for u,v,i,k in E} 
    
    #Cu= #number of slcies=Vms
    #U1,V1,K1,N1,Psi_u1,Phi_u1,B_u1,r_u1,tau_v1,tau_v2,sigma_v1,sigma_v2,b_v1,b_v2,L_v1,R_v1
	
	
    #return xmec,ymec,xdp,ydp,U,V,K,N,E,C_u,Psi_u,Phi_u,B_u,r_u,tau_v,sigma_v,b_v,L_v,R_v,C_ui,C_uik,PMEC,PDP
    return U,V,K,N,E,C_u,Psi_u,Phi_u,B_u,r_u,tau_v,sigma_v,b_v,L_v,R_v
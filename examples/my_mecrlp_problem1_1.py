from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
import numpy as np
import math
import datameclpps260421v1

class MyProblem(FloatProblem):
    
    def __init__(self, U1,V1,K1,N1,Psi_u1,Psi_u2,Phi_u1,Phi_u2,B_u1,B_u2,r_u1,tau_v1,tau_v2,sigma_v1,sigma_v2,b_v1,b_v2,L_v1,R_v1):
        super().__init__()
        #self.U = U # The array of MEC nodes
        #self.V = V # The array of Demand points
        #self.K = K # Redundancy
        #self.N = N # Number of shared demand points
        #self.C_u = C_u # The array of number of Slices in each MEC nodes
        #self.Psi_u = Psi_u # Psi_u MEC CPU capacity in MIPS
        #self.Phi_u = Phi_u # MEC RAM capacity in GB
        #self.B_u = B_u #MEC total bandwidth capacity in Mbps
        #self.r_u = r_u # Failure probability of the MEC u \in U
        #self.tau_v = tau_v # Processing demand of the demand point v in MIPS
        #self.sigma_v = sigma_v #Memory demand of the demand point v in GB = x8000 MBit
        #self.b_v = b_v # Bandwidth consumed by the demand point v in Mbps
        #self.L_v = L_v #Bound on the latency requirement of the demand point v in ms
        #self.R_v = R_v #Bound on the reliability requirement of the demand point v 
        self.x = {}
        self.y = {}
        self.w = {}
        self.psi_ui = {}
        self.phi_ui = {}
        
        self.U,self.V,self.K,self.N,self.E,self.C_u,self.Psi_u,self.Phi_u,self.B_u,self.r_u,self.tau_v,self.sigma_v,self.b_v,self.L_v,self.R_v,self.PMEC,self.PDP=datameclpps260421v1.make_data(U1,V1,K1,N1,Psi_u1,Psi_u2,Phi_u1,Phi_u2,B_u1,B_u2,r_u1,tau_v1,tau_v2,sigma_v1,sigma_v2,b_v1,b_v2,L_v1,R_v1)


        self.I = sum(self.C_u)
        #print(self.K)
        self.number_of_variables = len(self.V)*(self.K)*(self.I) + (self.K)*(self.I) + len(self.U)

        self.number_of_constraints = 11*self.I+9*len(self.V)+4*len(self.U)+2*self.K+self.K-3
       
        """ 
        for u in self.U:
            for i in range(self.C_u[u]):
                #const +=i
                #print(i)
                
                for k in range(self.K):
                    for v in self.V:
                        self.number_of_variables = len(self.U)*len(self.V)*self.K*self.C_u[u]+ len(self.U)*self.K*self.C_u[u] + len(self.U)+1
                        self.number_of_constraints =  len(self.U)*self.C_u[u]*2+len(self.V)*len(self.U)+1 """
        #print(const)
        self.number_of_objectives = 2
       
        self.lower_bound = [0.0 for _ in range(self.number_of_variables)]
        self.upper_bound = [1.0 for _ in range(self.number_of_variables)]
        #print(self.upper_bound)
        self.obj_directions = [self.MINIMIZE, self.MINIMIZE] # both objectives should be minimized
        self.obj_labels = ['N#Slices', 'N#MECs'] # objectives' name

    def evaluate(self, solution) -> FloatSolution:

        
        for u in self.U:
            
            if round(solution.variables[u])== 0:
                self.y[u] = 0
                for i in range(self.C_u[u]):
                    for k in range(self.K):
                        self.w[u,i,k] = 0
                        for v in self.V:
                         #self.x[u,v,i,k]= solution.variables[len(self.U)+len(self.U)*self.C_u[u]*self.K +u*self.C_u[u]*self.K*len(self.V)+i*self.K*len(self.V)+k*len(self.V)+v]
                            self.x[u,v,i,k] = 0
                            #self.z[u,v,i,k]= 0
            else:
                self.y[u] = round(solution.variables[u])
                for i in range(self.C_u[u]):
                    for k in range(self.K):
                        if round(solution.variables[len(self.U)+u*self.C_u[u]*self.K+i*self.K+k]) == 0:

                            self.w[u,i,k] = 0
                            for v in self.V:
                                #self.x[u,v,i,k]= solution.variables[len(self.U)+len(self.U)*self.C_u[u]*self.K +u*self.C_u[u]*self.K*len(self.V)+i*self.K*len(self.V)+k*len(self.V)+v]
                                self.x[u,v,i,k]= 0
                                #self.z[u,v,i,k]= 0
                        else:
                            self.w[u,i,k] = round(solution.variables[len(self.U)+u*self.C_u[u]*self.K+i*self.K+k])

                            for v in self.V:
                                #self.x[u,v,i,k]= solution.variables[len(self.U)+len(self.U)*self.C_u[u]*self.K +u*self.C_u[u]*self.K*len(self.V)+i*self.K*len(self.V)+k*len(self.V)+v]
                                self.x[u,v,i,k]= round(solution.variables[len(self.U)+(self.C_u[u]*self.K)+ u*self.C_u[u] +i*(self.K) +k*(len(self.V))+v] )
                                #self.z[u,v,i,k]= round(solution.variables[u*self.C_u[u]+i*self.K +k*(len(self.V))+v] )
        
        for u in self.U:
            solution.objectives[1] += self.y[u]
            for i in range(self.C_u[u]):
                 for k in range(self.K):
                     solution.objectives[0] += self.w[u,i,k]
            #print(f"y[{u}]= {self.y[u]} \n")

        
            
       
 
        

        #end defining the decision variables
        

        self.__evaluate_constraints(solution)




        return solution


    
       
    def __evaluate_constraints(self, solution: FloatSolution) -> None:
        constraints3,constraints3tmp = 0.0,0.0 #[0.0 for _ in range(self.number_of_constraints)]
        constraints4, constraints5,constraints6,constraints7   = 0.0,0.0,0.0,0.0
        """ constraints2 = 0.0
        constraints5 = 0.0
        constraints6 = 0.0
        constraints7 = 0.0
        constraints8 = 0.0
        constraints9, constraints10, constraints11, constraints14,constraints15, constraints16,constraints17,constraints18,constraints19,constraints20   =  0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0, 0.0, 0.0
        constraints16 = [0.0 for _ in range(len(self.U))]
        constraints17 = [0.0 for _ in range(len(self.U))]
        constraints18 = [0.0 for _ in range(len(self.U))] """
        self.z = {}
        for u in self.U:
            
            if round(solution.variables[u]) == 0:
                self.y[u] = 0
                for i in range(self.C_u[u]):
                    for k in range(self.K):
                        self.w[u,i,k] = 0
                        for v in self.V:
                         #self.x[u,v,i,k]= solution.variables[len(self.U)+len(self.U)*self.C_u[u]*self.K +u*self.C_u[u]*self.K*len(self.V)+i*self.K*len(self.V)+k*len(self.V)+v]
                            self.x[u,v,i,k]= 0
                            self.z[u,v,i,k]= 0
            else:
                self.y[u] = round(solution.variables[u])
                for i in range(self.C_u[u]):
                    for k in range(self.K):
                        if round(solution.variables[len(self.U)+u*self.C_u[u]*self.K+i*self.K+k]) == 0:

                            self.w[u,i,k] = 0
                            for v in self.V:
                                #self.x[u,v,i,k]= solution.variables[len(self.U)+len(self.U)*self.C_u[u]*self.K +u*self.C_u[u]*self.K*len(self.V)+i*self.K*len(self.V)+k*len(self.V)+v]
                                self.x[u,v,i,k]= 0
                                self.z[u,v,i,k]= 0
                        else:
                            self.w[u,i,k] = round(solution.variables[len(self.U)+u*self.C_u[u]*self.K+i*self.K+k])

                            for v in self.V:
                                #self.x[u,v,i,k]= solution.variables[len(self.U)+len(self.U)*self.C_u[u]*self.K +u*self.C_u[u]*self.K*len(self.V)+i*self.K*len(self.V)+k*len(self.V)+v]
                                self.x[u,v,i,k]= round(solution.variables[len(self.U)+(self.C_u[u]*self.K)+ u*self.C_u[u] +i*(self.K) +k*(len(self.V))+v])
                                self.z[u,v,i,k]= round(solution.variables[len(self.U)+(self.C_u[u]*self.K)+ u*self.C_u[u] +i*(self.K) +k*(len(self.V))+v])

            """ for i in range(self.C_u[u]):
                for k in range(self.K):
                    self.w[u,i,k] = round(solution.variables[u*self.C_u[u]+i*self.K+k])
                    for v in self.V:
                         #self.x[u,v,i,k]= solution.variables[len(self.U)+len(self.U)*self.C_u[u]*self.K +u*self.C_u[u]*self.K*len(self.V)+i*self.K*len(self.V)+k*len(self.V)+v]
                         self.x[u,v,i,k]= round(solution.variables[u*self.C_u[u]+i*self.K +k*(len(self.V))+v] )
                         self.z[u,v,i,k]= round(solution.variables[u*self.C_u[u]+i*self.K +k*(len(self.V))+v] )
        for u in self.U:
            if self.y[u] == 0:
                for i in range(self.C_u[u]):
                    for k in range(self.K):
                        self.w[u,i,k] = 0
                        for v in self.V:
                             #self.x[u,v,i,k]= solution.variables[len(self.U)+len(self.U)*self.C_u[u]*self.K +u*self.C_u[u]*self.K*len(self.V)+i*self.K*len(self.V)+k*len(self.V)+v]
                            self.x[u,v,i,k]= 0
                            self.z[u,v,i,k]= 0 """


        
        constraints3 = [0.0 for u in self.U for i in range(self.C_u[u])]
        constraints3tmp = [0.0 for u in self.U for i in range(self.C_u[u])]

        constraints4 = [0.0 for u in self.U for i in range(self.C_u[u])]
        constraints4tmp = [0.0 for u in self.U for i in range(self.C_u[u])]
        
        constraints5 = [0.0 for v in self.V for u in self.U]
        constraints5tmp = [0.0 for v in self.V for u in self.U]
        
        constraints6 = [0.0 for v in self.V]
        constraints6tmp = [0.0 for v in self.V]

        constraints7 = [0.0 for u in self.U for i in range(self.C_u[u]) for k in range(2,self.K)]
        constraints7tmp = [0.0 for u in self.U for i in range(self.C_u[u]) for k in range(2,self.K)]

        constraints8 = [0.0 for v in self.V]
        constraints8tmp = [0.0 for v in self.V]

        constraints9 = [0.0  for v in self.V for u in self.U for i in range(self.C_u[u])]
        constraints9tmp = [0.0  for v in self.V for u in self.U for i in range(self.C_u[u])]

        constraints10 = [0.0  for v in self.V for u in self.U for i in range(self.C_u[u])]
        

        constraints11 = [0.0  for u in self.U for i in range(self.C_u[u]) for k in range(self.K)]

        self.psi_ui = {(u,i):0.0  for u in self.U for i in range(self.C_u[u])}
        self.psi_uitmp = {(u,i):0.0  for u in self.U for i in range(self.C_u[u])}
        self.phi_uitmp = {(u,i):0.0  for u in self.U for i in range(self.C_u[u])}

        self.taspsi_ui = {(u,i):1/(min(self.Psi_u))  for u in self.U for i in range(self.C_u[u])}





        constraints12 = [0.0  for u in self.U for i in range(self.C_u[u])]
        constraints13 = [0.0  for u in self.U for i in range(self.C_u[u])]

        constraints16 = [0.0  for u in self.U ]
        constraints16tmp = [0.0  for u in self.U ]

        constraints17 = [0.0  for u in self.U ]
        constraints17tmp = [0.0  for u in self.U ]

        constraints18 = [0.0  for u in self.U]
        constraints18tmp = [0.0  for u in self.U]

        constraints19 = [0.0  for v in self.V for u in self.U for i in range(self.C_u[u])]
        constraints19tmp1=[0.0  for v in self.V for u in self.U for i in range(self.C_u[u])]
        constraints19tmp2 =[0.0  for v in self.V for u in self.U for i in range(self.C_u[u])]

        constraints20 = [0.0  for v in self.V]
        constraints20tmp = [0.0  for v in self.V]

        constraints21 = [0.0  for u in self.U for i in range(self.C_u[u]) for k in range(self.K)]
        constraints22 = [0.0  for u in self.U for i in range(self.C_u[u]) for k in range(self.K)]

        constraints23 = [0.0  for u in self.U for i in range(self.C_u[u]) for k in range(self.K)]
        constraints24 = [0.0  for u in self.U for i in range(self.C_u[u]) for k in range(self.K)]


        """ print(constraints18tmp)

        
        

        print (len(constraints3))
        print(f"total de Cu {self.I}") 
        print (len(self.x))
        print (len(self.w))
        print (len(self.y)) """
        
        #constraint 3 and 4
        #sum3,sum4 = 0.0,0.0
        for u in self.U:
            for i in range(self.C_u[u]):
                for v in self.V:
                    constraints3tmp[i] += self.x[u,v,i,0]
                    constraints4tmp[i] += self.x[u,v,i,1]
                    #constraints4tmp[u*self.C_u[u]+i] += self.x[u,v,i,1]
                    #sum4 = sum4 + self.x[u,v,i,1]
        
        for u in self.U:
            for i in range(self.C_u[u]):
                constraints3[i] = 1 - constraints3tmp[i]
                constraints4[i] = self.N - constraints4tmp[i]
                #constraints4[i] = self.N - constraints4tmp[u*self.C_u[u]+i]
        
        #constraint 5
        for v in self.V:
            for u in self.U:
                for i in range(self.C_u[u]):
                    for k in range(self.K):
                        constraints5tmp[v*len(self.U)+u] += self.x[u,v,i,k]
                constraints5[v*len(self.U)+u] = 1 - abs(constraints5tmp[v*len(self.U)+u])

        """ for v in self.V:
            for u in self.U:
                constraints5[v*len(self.U)+u] = 1 - abs(constraints5tmp[v*len(self.U)+u]) """
        
        #Constraint 6
        for v in self.V:
            for u in self.U:
                for i in range(self.C_u[u]):
                    constraints6tmp[v] += self.x[u,v,i,0]
            constraints6[v] = 1 - constraints6tmp[v]
        
        """ for v in self.V:
            constraints6[v] = 1 - constraints6tmp[v] """
        
        #Constraint 7 

        for u in self.U:
            for i in range(self.C_u[u]):
                for k in range(2,self.K):
                    for v in self.V:
                        constraints7tmp[i*self.K+k] += self.x[u,v,i,k]
                    constraints7[i*self.K+k] = len(self.V) - constraints7tmp[i*self.K+k]
        
        """ for u in self.U:
            for i in range(self.C_u[u]):
                for k in range(2,self.K):
                    constraints7[i*self.K+k] = len(self.V) - constraints7tmp[i*self.K+k] """
        
        #Constraint 8
        for v in self.V:
            for u in self.U:
                for i in range(self.C_u[u]):
                    for k in range(1,self.K):
                        constraints8tmp[v] = self.x[u,v,i,k]
                        
        
        for v in self.V:
            for u in self.U:
                for i in range(self.C_u[u]):
                    for k in range(self.K):
                        constraints8[v] =-self.K +abs(constraints8tmp[v])

        #Constraint 9
        for v in self.V:
            for u in self.U:
                for i in range(self.C_u[u]):
                    for k in range(self.K):
                        constraints9tmp[v*len(self.U)+u*self.C_u[u]] += self.x[u,v,i,k]
        for v in self.V:
            for u in self.U:
                for i in range(self.C_u[u]):
                    for k in range(self.K):
                         constraints9[v*len(self.U)+u*self.C_u[u]] = 1 - constraints9tmp[v*len(self.U)+u*self.C_u[u]] 
        
        # Constraint 10 and 11
        for v in self.V:
            for u in self.U:
                for i in range(self.C_u[u]):
                    for k in range(self.K):
                        constraints10[v*len(self.U)+u*self.C_u[u]+i*self.K+k] = -self.x[u,v,i,k]+self.w[u,i,k]
        
        #Constraints11
        for u in self.U:
                for i in range(self.C_u[u]):
                    for k in range(self.K):
                        constraints11[u*self.C_u[u]+i*self.K +k] = self.y[u] - self.w[u,i,k]

        #Constraint 12 and 13
        for u in self.U:
            for i in range(self.C_u[u]):
                self.psi_uitmp[u,i] = max(self.tau_v[v]*self.x[u,v,i,k] for v in self.V  for k in range(1,self.K) )
                self.phi_uitmp[u,i] = max(self.sigma_v[v]*self.x[u,v,i,k] for v in self.V  for k in range(1,self.K) )
        #Constraint 14 and 15
        for u in self.U:
            for i in range(self.C_u[u]):
                for v in self.V:
                    self.psi_uitmp[u,i] = self.tau_v[v]*self.x[u,v,i,0]
                    self.phi_uitmp[u,i] = self.sigma_v[v]*self.x[u,v,i,0]

        #Constraint 16 and 17
        for u in self.U:
            for i in range(self.C_u[u]):
                constraints16tmp [u] +=   self.psi_uitmp[u,i]
            constraints16[u] = self.Psi_u[u] - constraints16tmp[u] 

        """ for u in self.U:
            for i in range(self.C_u[u]):
                constraints16[u] = self.Psi_u[u] - constraints16tmp[u] 
 """
        for u in self.U:
            for i in range(self.C_u[u]):
                constraints17tmp [u] += self.phi_uitmp[u,i]
            constraints17[u] = self.Phi_u[u] - constraints17tmp[u] 

        """ for u in self.U:
            for i in range(self.C_u[u]):
                constraints17[u] = self.Phi_u[u] - constraints17tmp[u] 
 """
        #Constraint 18
        for u in self.U:
            for i in range(self.C_u[u]):
                for v in self.V:
                    for k in range(self.K):
                        constraints18tmp[u] += self.b_v[u,v,i,k]*self.x[u,v,i,k]
            constraints18[u] = self.B_u[u] - constraints18tmp[u]

        """ for u in self.U:
            constraints18[u] = self.B_u[u] - constraints18tmp[u] """
        
        #constraint 19

        for v in self.V:
            for u in self.U:
                for m in self.U:
                    if u!=m:
                        for i in range(self.C_u[m]):
                            for k in range(self.K):
                              constraints19tmp2[v*len(self.U)+u*self.C_u[m]+i] +=self.x[m,v,i,k]*math.log(1-self.r_u[m]) 
                    else:
                        for i in range(self.C_u[u]):
                            constraints19tmp1[v*len(self.U)+u*self.C_u[u]+i] +=self.x[u,v,i,0]*math.log(1-self.r_u[u]) 
            constraints19[v] =-constraints19tmp1[v] -constraints19tmp2[v] +math.log(1-self.R_v[v])

        
        #Constraint 20
        for u in self.U:
            for i in range(self.C_u[u]):
                self.psi_ui[u,i]*self.taspsi_ui[u,i] == 1
        
        for v in self.V:
            for u in self.U:
                for i in range(self.C_u[u]):
                    for k in range(self.K):
                        self.z[u,v,i,k] = self.taspsi_ui[u,i]*self.x[u,v,i,k]

        for v in self.V:
            for u in self.U:
                for i in range(self.C_u[u]):
                    for k in range(self.K):
                        constraints20tmp [v] += self.z[u,v,i,k]*self.tau_v[v]
            constraints20[v] = self.L_v[v] -constraints20tmp[v]
        
        for v in self.V:
            for u in self.U:
                for i in range(self.C_u[u]):
                    for k in range(self.K):
                        constraints21[u*self.C_u[u]+i*self.K+k] += -self.z[u,v,i,k]+(1/min(self.Psi_u))*self.x[u,v,i,k]

        for v in self.V:
            for u in self.U:
                for i in range(self.C_u[u]):
                    for k in range(self.K):
                        constraints22[u*self.C_u[u]+i*self.K+k] = self.z[u,v,i,k]-(1/max(self.Psi_u))*self.x[u,v,i,k]

        """ for v in self.V:
            for u in self.U:
                for i in range(self.C_u[u]):
                    for k in range(self.K):
                        constraints20tmp[v] += (self.tau_v/self.psi_uitmp[u,i])*self.x[u,v,i,k]
            constraints20[v] = self.L_v[v] - constraints20tmp[v] """

        #Constraint 24

        for v in self.V:
            for u in self.U:
                for i in range(self.C_u[u]):
                    for k in range(self.K):
                        constraints23[u*self.C_u[u]+i*self.K+k] += -self.z[u,v,i,k]+self.taspsi_ui[u,i]+(1/max(self.Psi_u))*(1-self.x[u,v,i,k])
                        constraints24[u*self.C_u[u]+i*self.K+k] += self.z[u,v,i,k]-self.taspsi_ui[u,i]+(1/min(self.Psi_u))*(1-self.x[u,v,i,k])

        #solution.constraints = constraints2
        solution.constraints = constraints3
        solution.constraints = constraints4
        solution.constraints = constraints5
        solution.constraints = constraints6
        solution.constraints = constraints7
        solution.constraints = constraints8
        solution.constraints = constraints9
        solution.constraints = constraints10
        solution.constraints = constraints11
        solution.constraints = constraints16
        solution.constraints = constraints17
        solution.constraints = constraints18
        solution.constraints = constraints19
        solution.constraints = constraints20
        solution.constraints = constraints21
        solution.constraints = constraints22
        solution.constraints = constraints23
        solution.constraints = constraints24
        


    def get_name(self):
        return 'My_MECRLP_problem'
#from jmetal.algorithm.multiobjective.gde3 import GDE3
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
#from jmetal.algorithm.multiobjective.smpso import SMPSO
from jmetal.core.quality_indicator import *
from jmetal.lab.experiment import Experiment, Job, generate_summary_from_experiment
from jmetal.operator import PolynomialMutation, SBXCrossover
from my_mecrlp_problem1_1 import *
#import datameclpps260421
from jmetal.util.archive import CrowdingDistanceArchive
from jmetal.util.termination_criterion import StoppingByEvaluations
import sys





def configure_experiment(problems: dict, n_run: int):
    jobs = []
    max_evaluations = 5

    for run in range(n_run):
        for problem_tag, problem in problems.items():
            jobs.append(
                Job(
                    algorithm=NSGAII(
                        problem=problem,
                        population_size=10,
                        offspring_population_size=10,
                        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables,
                                                    distribution_index=20),
                        crossover=SBXCrossover(probability=1.0, distribution_index=20),
                        termination_criterion=StoppingByEvaluations(max_evaluations)
                    ),
                    algorithm_tag='NSGAII',
                    problem_tag=problem_tag,
                    run=run,
                )
            )

    return jobs

if __name__ == '__main__':
    """ #U1 = {10}
    U1 = {5}
    #V1 = {10}
    V1 = {10}
    K1 = {3}
    #K1 = {3,4}
    N1 ={2}
    #N1 ={5,10,15,20}
    L_v1 ={0.001}
    #r_u1 = {0.001%,0.1%,1.0%,5.0%}
    r_u1 = {0.00001}
    R_v1 ={0.99999}"""

    countu =0.0
    countuik =0.0
    countv =0.0

    #U1 = {int(sys.argv[1])}
    U1 = {int(sys.argv[1])}
    #V1 = {int(sys.argv[2])}
    V1 = {int(sys.argv[2])}
    K1 = {2}
    #K1 = {3,4}
    N1 ={0}
    #N1 = {int(sys.argv[3])}
    L_v1 ={0.001,0.005,0.01,0.02}
    #r_u1 = {0.001%,0.1%,1.0%,5.0%}
    r_u1 = {0.00001,0.001,0.01,0.05}
    R_v1 ={0.99999,0.9999,0.999,0.99}

    """
    countu =0.0
    countuik =0.0
    countv =0.0 """
 


    #U ={}
    #V ={}
    #K={}
    #N={}
    #front=""
    #C_u,Psi_u, Phi_u, B_u, r_u, tau_v, sigma_v, b_v, L_v,R_v,PMEC,PDP,E,sol1 = {},{},{},{},{},{} ,{},{},{},{},{},{},{},{}
    #mp = MyProblem(30,U,V,K,N,E,C_u,Psi_u,Phi_u,B_u,r_u,tau_v,sigma_v,b_v,L_v,R_v,PMEC,PDP)
    mp =""
    for u in U1:
        for n in N1:
            for v in V1:
                for lv in L_v1:
                    for ru in r_u1:
                        for rv in R_v1:
                            for k in K1:
                                #filename = '/data3/'+str(u)+"/"+str(n)+"/"+str(v)+"/"+str(lv)+"/"+str(ru)+"/"+str(rv)+"/"+str(k)+"/outputsolution.tsv"
                                #filename1 = '/data3/'+str(u)+"/"+str(n)+"/"+str(v)+"/"+str(lv)+"/"+str(ru)+"/"+str(rv)+"/"+str(k)+"/teste_.tsv"
                                
                                
                                mp = MyProblem(u,v,k,n,2600,4800 ,16000 ,64000,400,800,ru,100,400,612,1740,10,40,lv,rv)

                                jobs = configure_experiment(problems={'MECRLP': mp}, n_run=2)
                                output_directory = '/data11/'+str(u)+"/"+str(n)+"/"+str(v)+"/"+str(lv)+"/"+str(ru)+"/"+str(rv)+"/"+str(k)+"/"
                                experiment = Experiment(output_dir=output_directory, jobs=jobs)
                                experiment.run()
                                """ for i in range(1,2):
                                    experiment.run()
                                    #front = get_non_dominated_solutions(experiment.get_result()) 
                                    #plot_front = Plot(title='Pareto front approximation', axis_labels=['N# Slices', 'N. MEC'])
                                    #plot_front.plot(front, label='NSGAII', filename='My-MECLRP', format='pdf')
                                    for u1 in range(u):
                                        countu += mp.y[u1]
                                        for i1 in range(mp.C_u[u1]):
                                            for k1 in range(k):
                                                countuik += mp.w[u1,i1,k1]
                                                for v1 in range(v):
                                                    countv += mp.x[u1,v1,i1,k1]
                                    
                                    with open(filename1, 'a') as f1:
                                        f1.write(str(f"{countuik}"))
                                        f1.write("\t")
                                        f1.write(str(f"{countu}"))
                                        f1.write("\t")
                                        f1.write(str(f"{countv}"))
                                        f1.write("\n")

                                with open(filename,"a") as f:
                                    for u1 in range(u):
                                        f.write(str(f" y[{u1}]={mp.y[u1]}"))
                                        f.write("\n")
                                        for i1 in range(mp.C_u[u1]):
                                            for k1 in range(k):
                                                f.write("\t")
                                                f.write(str(f" w[{u1,i1,k1}]={mp.w[u1,i1,k1]}"))
                                                f.write("\n")
                                                for v1 in range(v):
                                                    f.write("\t")
                                                    f.write(str(f" x[{u1,v1,i1,k1}]={mp.x[u1,v1,i1,k1]}"))
                                                    f.write("\n")
                                    f.write("\n")
                                    f.write(str(f" Total of opened MEC ={countu}")) 
                                    f.write("\n")
                                    f.write(str(f" Total of opened slices ={countuik}"))
                                    f.write("\n")
                                    f.write(str(f" Total of served demand points ={countv}")) """
                                
                                

                                    
                                    



    # Configure the experiments
    #mp = MyProblem(U,V,K,N,C_u,Psi_u,Phi_u,B_u,r_u,tau_v,sigma_v,b_v,L_v,R_v)
    #jobs = configure_experiment(problems={'MECRLP': mp}, n_run=5)

    # Run the study
    #output_directory = 'data1'
    #experiment = Experiment(output_dir=output_directory, jobs=jobs)
    #experiment.run()
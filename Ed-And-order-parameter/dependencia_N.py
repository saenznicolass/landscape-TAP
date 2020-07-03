
import matplotlib
matplotlib.use('Agg')

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import collections
import numpy as np
import numpy.matlib
import time
import random
import os
import errno
import csv

tolerance = 0.00000000000000001
temp=0.1
beta = 1./temp
mean = 0


def acoplamientos(N):
    sym=numpy.zeros(shape=(N,N,N))
    standard_deviation = np.sqrt(3)/N
    for i in range (N):
        for j in range (N):
            for k in range (N):
                if ((i-j)*(i-k)*(j-k)!=0): 
#                    J=np.random.normal(loc = mean, scale = standard_deviation)
                    J=np.random.randn()*standard_deviation
#                    print("---------tensor de magnetizaciones---------")
#                    print(i,j,k)
                    sym[i,j,k]=J
                    sym[i,k,j]=J
                    sym[j,i,k]=J
                    sym[j,k,i]=J
                    sym[k,j,i]=J
                    sym[k,i,j]=J
                else:
                    sym[i,j,k] = 0
                    sym[i,k,j] = 0
                    sym[j,i,k] = 0
                    sym[k,j,i] = 0

    #print(sym)
    return sym

def elem_rep(lis):
    count=collections.Counter(lis)
    obj=[]
    rep=[]
    for key, value in count.items():
        obj.append(key)
        rep.append(value)
    return obj, rep
        

def mag_updated(interaction,magnetization,N,beta):
    error=0
    #print("-----------interacion matrix---------")
    #print(interaction)
    for i in range (N):
        tens_mag=0
        ons=0
        for j in range (N):
            Ed_And=0
            for k in range(N):
                tens_mag+=interaction[i,j,k]*magnetization[j,1]*magnetization[k,1]
                #print("---------i en el tensor-------")
                #print("i=",i,"j=",j, magnetization[j,1],"k=",k, magnetization[k,1],"tensor= ",interaction[i,j,k],"suma= ",tens_mag)
                Ed_And+=(interaction[i,j,k])**2*(1-(magnetization[k,0])**2)
            ons+=(1-(magnetization[j,0])**2)*Ed_And
        insidetanh=(beta*tens_mag/2)-(beta**2*ons*magnetization[i,0])
        magnetization[i,2]=np.tanh(insidetanh)
        error+=(magnetization[i,2]-magnetization[i,1])**2
    error=np.sqrt(error)/N
    return magnetization, error
    
def energy_entropy(interaction,magnetization,N,T):
    ener=0
    q=0
    q4=0
    entr=0
    onsager=0
    beta=1/T
    for i in range(N):
        for j in range(N):
            for k in range(N):
                ener+=interaction[i,j,k]*magnetization[i,2]*magnetization[j,2]*magnetization[k,2]
                onsager+= (interaction[i,j,k])**2*(1-(magnetization[i,2])**2)*(1-(magnetization[j,2])**2)*(1-(magnetization[i,2])**2)
        q+=(magnetization[i,2])**2
        q4+=(magnetization[i,2])**4
        if magnetization[i,2]==1 or magnetization[i,2]==-1:
            entr_each_i=0
        else:
            entr_each_i=((1+magnetization[i,2])/2*np.log(np.abs(1+magnetization[i,2]/2)))+((1-magnetization[i,2])/2*np.log(np.abs(1-magnetization[i,2]/2)))
        entr+=entr_each_i
    #ener=-1*ener/(6*N)
    ener=(-1)*ener/(N)
    entr=entr/N
    onsager=onsager/N
    q=q/N
    q4=q4/N
    ener_free=(ener/6)+(entr*T)-(beta*onsager/4)
    return ener,ener_free, entr, onsager,q,q4
                
def over_time(ticks,interaction,N,beta,magne,n):
    if n==0:
        mag=np.random.random((N, 3))*2-1
    else:
        mag=magne
    fixed=0
    number_fixed_single_mag=0
    energy_free=0
    ener_int=0
    counter=0
    q=0
    q4=0
    T=1/beta
    for i in range (ticks):
        magnetization,error=mag_updated(interaction,mag,N,beta)
        if error<=tolerance:
            fixed=True
            break
            
        else:
            mag[:, 0:1]=mag[:, 1:2]
            mag[:, 1:2]=magnetization[:, 2:3]
            mag=mag
            counter+=1
            continue
        
    if fixed is True:
        number_fixed_single_mag+=1
        ener_int,energy_free, entropy, onsager, q,q4=energy_entropy(interaction,mag,N,T)
    ener_int=ener_int/6
    magnet=mag
    return ener_int,energy_free, number_fixed_single_mag, q,q4,magnet, fixed


def over_magnetizations(mag_unos,ticks,interaction,N,beta):
    number_fixed_J=0
    energytot_free=[]
    q_tot=[]
    q4_tot=[]
    ener_interaction=[]

    for elem in mag_unos:
        magneti=elem
        ener_int,energy_free,number_fixed_single_mag,q,q4,magter,fixed=over_time(ticks,interaction,N,beta,magneti,1)
        if energy_free!=0:
            energytot_free.append(energy_free)
            ener_interaction.append(ener_int)
            q_tot.append(q)
            q4_tot.append(q4)
            number_fixed_J+=number_fixed_single_mag
        else:
            continue
            
    return ener_interaction,energytot_free,q_tot,number_fixed_J,q4_tot
    
    
def ener_igua(ene_mon,ene_tap):
    a=list(set(ene_mon).intersection(ene_tap))
    return(a)



def hessian(interaction,magnetization,N,beta):
    hess=numpy.zeros(shape=(N,N))
    tens_mag=0
    Ed_And=0
    for i in range (N):
        for j in range (N):
            if (i-j)!=0:
                for k in range(N):
                    tens_mag+=interaction[i,j,k]*magnetization[k]
                    Ed_And+=(interaction[i,j,k])**2*(1-(magnetization[k])**2)
                hess[i,j]=(-tens_mag)-(beta*Ed_And*magnetization[i]*magnetization[j])
            else:
                hess[i,j]=1/((1-magnetization[i]**2)*beta)
                # print("-------------magnetizacion que va a la diagonal del hessiano------------")
                # print(magnetization[i])
                # print("-------------valores de la diagonal del hesiano-------------")
                # print(hess[i,j])
    return hess



def initial_iterative(ticks,interaction,N,Temps):
    magne=[1]
    fixed=True
    te=0
    energias=[]
    hessian_eig=[]
    list_t_lim=[]
    list_q=[]
    while fixed==True:
        
        for i in range(len(Temps)):
            
            beta=1/Temps[i]
            ener_int,energy_free, number_fixed_single_mag, q,q4,magnet,fix=over_time(ticks,interaction,N,beta,magne,i)
            # print("---------------enegia libre----------------")
            # print(energy_free)
            if fix !=True:
                fixed=fix
                break
            else:
                # print("i=",i,"T=", Temps[i],q)
                
                if i==len(Temps)-1:
                    magne=magnet
                    te=Temps[i]
                    fixed=False
                    energias.append(ener_int)
                    list_t_lim.append(Temps[i])
                    list_q.append(q)
                else:
                    magne=magnet
                    te=Temps[i]
                    energias.append(ener_int)
                    list_t_lim.append(Temps[i])
                    list_q.append(q)
                # print("magnetization:")
                # print(magne[:,2])
            hess=hessian(interaction,magne[:,2],N,beta)
            # print("-----------hessian--------------")
            # print(hess)
            eigen=np.linalg.eigvals(hess)
            hessian_eig.append(eigen)
            # print("-------------eigen-----------")
            # print(eigen)
#    print("last T=", te)        
    return te, energias, hessian_eig,list_t_lim, list_q


##########################################################
##########################################################
########## de aqui para abajo era la solucion que ########
########## reciclaba el mismo vector de magnetizacion ####
##########################################################
##########################################################
##########################################################

    
# multi=np.linspace(0.15,0.8,50)
# interaction=acoplamientos(20)
# # for i in range(20):

# #     te,en,eigen=initial_iterative(100, interaction, 20, multi)
# #     # FIG=plt.figure()
# #     # plt.plot(t,energiamonte, 'go--', linewidth=1, markersize=1, label = "Energy")
# #     # plt.xlabel("Time")
# #     # plt.ylabel("E")
# #     # plt.title("E(t)",fontsize=20)
# #     # plt.legend()
# #     # plt.grid()
# #     # FIG.savefig('Energy(t).png')
  

# list_list_ene=[]
# list_list_temp=[]
# list_q_general=[]

# for i in range(1000):
#     te=None
#     en=[]
#     eigen=None
#     list_t=[]
#     list_q=[]
#     try:
#         te,en,eigen,list_t,list_q=initial_iterative(150, interaction, 20, multi)
#     except: 
#         continue
#     finally:
#         if len(en)==0 :
#             pass
#         else:
#             # print("---lista de t-----")
#             # print(list_t)
#             # print("----energias")
#             # print(en)
#             list_list_ene.append(en)
#             list_list_temp.append(list_t)
#             list_q_general.append(list_q)
#             FIG=plt.figure()
#             plt.plot(list_t,en, label='energy')
#             plt.xlabel("Temperature")
#             plt.ylabel("energy")
#             plt.title('ener_vs_T')
#             plt.legend()
#             plt.grid()
#             FIG.savefig('energy_vs_T_{}.png'.format(i))
# FIG2=plt.figure()
# plt.xlabel("Temperature")
# plt.ylabel("energy")
# plt.title('ener_vs_T')
# for i in range(len(list_list_ene)):
#     plt.plot(list_list_temp[i],list_list_ene[i])
# #plt.legend()
# plt.grid()
# FIG2.savefig('energy_vs_T_master.png')
    
# q_a_plot = [item for sublist in list_q_general for item in sublist]
# t_a_plot_in_q = [item for sublist in list_list_temp for item in sublist]
# e_a_plot_in_q = [item for sublist in list_list_ene for item in sublist]
# FIG3=plt.figure()
# plt.xlabel("Temperature")
# plt.ylabel("q_EA")
# plt.title('q_EAvs_T')
# plt.scatter(t_a_plot_in_q,q_a_plot)
# plt.grid()
# FIG3.savefig('q_EA_vs_T.png')

# FIG4=plt.figure()
# plt.xlabel("Energy")
# plt.ylabel("q_EA")
# plt.title('q_EA_vs_T')
# plt.scatter(e_a_plot_in_q,q_a_plot)
# plt.grid()
# FIG4.savefig('q_EA_vs_energy.png')




###########################################################
###########################################################
########## Aca comienza lo del reporte a entregar #########
###########################################################
###########################################################


N=20
n=20
mag_list=np.load("list_qea_inter_free_N={}_array_{}.npy".format(N,n-1))

t1=time.time()
mat_interaction=numpy.load("matriz_J.npy")
temperatura=0.3
beta=1/temperatura


energy_list, energytot_free, q_list, number_fixed_J, q4_tot = over_magnetizations(mag_list,500,mat_interaction,N,beta)

with open("qea_interaction_free_{}_t={}.csv".format(n,temperatura), "a", newline='') as f:
    wr = csv.writer(f)
    wr.writerows(zip(q_list,energy_list,energytot_free))


t2=time.time()
print(t2-t1)
tim=t2-t1
with open("time.csv", "a", newline='') as f:
    wr = csv.writer(f)
    wr.writerows(zip([temperatura],[tim]))

##############esto funciona para escribir columnas cuando se tienen lista de listas###########
# with open("qea_vs_energy_aveplane_{}.csv".format(n), "a", newline='') as f:

#     for i in range(len(q_list_list)):
#         wr = csv.writer(f)
#         wr.writerows(zip(q_list_list[i],energy_int_list_list[i]))
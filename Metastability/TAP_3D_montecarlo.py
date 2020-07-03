
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
import pandas as pd
import csv

tolerance = 0.00000000000001
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
#    ener=float("{0:.10f}".format((-1)*ener/(N)))
    ener=(-1)*ener/(N)
#    entr=entr/N
    entr=float("{0:.10f}".format(entr/N))
    onsager=onsager/N
    q=q/N
    q4=q4/N
    ener_free=(ener/6)+(entr*T)-(beta*onsager/4)
    return ener,ener_free, entr, onsager,q,q4
                
def over_time(ticks,interaction,N,beta):
    mag=np.random.random((N, 3))*2-1
    fixed=0
    number_fixed_single_mag=0
    energy_free=0
    ener_int=0
    counter=0
    q=0
    q4=0
    T=1/beta
    #print("----------J initial--------")
    #print(interaction)
    #print("----------mag initial--------")
    #print(mag)
    for i in range (ticks):
        magnetization,error=mag_updated(interaction,mag,N,beta)
        #print("----------vector de tiempo actual de la magnetizacion en cada tick--------------")
        #print(magnetization)
        if error<=tolerance:
            #print("fixed point", counter)
            fixed=True
            #print("----------mag updated last--------")
            #print(mag)
            break
            
        else:
            mag[:, 0:1]=mag[:, 1:2]
            mag[:, 1:2]=magnetization[:, 2:3]
            mag=mag
            #print("----------magnetizacion en cada tick--------")
            #print(mag)
#            print("----------mag updated--------")
#            print(mag)
            counter+=1
            continue
        
    if fixed is True:
        number_fixed_single_mag+=1
        ener_int,energy_free, entropy, onsager, q,q4=energy_entropy(interaction,mag,N,T)
#        print("magnetizacion equilibrio")
#        print(mag[:,2:3])
        #print(energy,number_fixed_single_mag)
        #print("q=",q)
        #print("punto encontrado")
    ener_int=float("{0:.10f}".format(ener_int/6))
    return ener_int,energy_free, number_fixed_single_mag, q,q4


def over_magnetizations(number_magnetizations,ticks,interaction,N,beta):
    number_fixed_J=0
    energytot_free=[]
    q_tot=[]
    q4_tot=[]
    ener_interaction=[]
    for i in range (number_magnetizations):
        #print("-----numero de matriz de magnetizacion:",i,"--------")
        ener_int,energy_free,number_fixed_single_mag,q,q4=over_time(ticks,interaction,N,beta)
        if energy_free!=0:
            energytot_free.append(energy_free)
            ener_interaction.append(ener_int)
            q_tot.append(q)
            q4_tot.append(q4)
            number_fixed_J+=number_fixed_single_mag
        else:
            continue
            
    return ener_interaction,energytot_free,q_tot,number_fixed_J,q4_tot
    
############ dos posibilidades para muestreo, uno, hacer muestreo por T, o sea, cambiando T y la otra, hacerlo cambiando J.    
# la que está ahrita va a ser para varias temperaturas, aquella de TAP3D_gasta histograma, esta hecha para varias J, por ende, pude hacer histogramas para una sola J a una sola T
#esta misma, va a ser el recorrido dobre las temperaturas
def muestreo(number_mag,ticks,N,Temps,interaction):
    total_fixed=[]
    q_average_list=[]
    q_list_list=[]
    q4_list_list=[]
    energy_distr=[]
    energy_free=[]
    energy_int_list_list=[]
    real_av_list=[]
    mat_inte=interaction
    for t in Temps:
        #print("--------sample---------", i)
#        mat_inte=acoplamientos(N)
        #print("temp=",t)
        beta=1/t
        ener_int,energyto_free,q_list,number_fixed,q4_tot=over_magnetizations(number_mag,ticks,mat_inte,N,beta)
        energy_int_list_list.append(ener_int)
        total_fixed.append(number_fixed)
        
        q_list_list.append(q_list)
        q_average=np.average(q_list)
        q_average_list.append(q_average)
        energy_free.append(energyto_free)
#        print("-----------------energyto-----------------",N)
#        print(energyto_free)
#        energylist=list((-1/N)*(np.array(energyto_free))**2)
        energylist=list((-1*beta)*(np.array(energyto_free)))
#        print("--------------lista de nuevo--------------------")
#        print(energylist)
        listagain=np.exp(energylist)
        energy_distr.append(listagain)
        q4_list_list.append(q4_tot)
        numerator=list((np.array(q_list))*(np.array(listagain)))
        denominator=sum(listagain)
        real_av=(sum(numerator))/denominator
        real_av_list.append(real_av)
    #return energyto,q_list_, q_average,total_fixed
    #return ("lista de lista de q= ",q_list_list, "lista de q promediados=",q_average_list,"lista de el total de soluciones encontradas=",total_fixed)
    return energy_int_list_list, q_list_list, q_average_list, total_fixed, energy_distr, q4_list_list, energy_free, real_av_list

def q_theory(T,T_c):
    q1=1-(np.sqrt(2/(3*np.pi))*(T/T_c))
    q2=-1/2*(1-(T_c/T)**2)
    return q1,q2


def montecarlo(interaction,NN,Temp,time):
    magnetization=np.random.random((NN, 3))*2-1
    energ_list=[]
    t=[]
    t.append(0)
    b=1/Temp
    ener,ener_free, entr, onsager,q,q4=energy_entropy(interaction,magnetization,NN,Temp)
    energ_list.append(ener)
    ener_rep=[]
    num_rep=[]
    pos_rep=[]
    for i in range (time):
        ir=random.randint(0, NN-1)
        elem_magnetization=np.random.random()*2-1
        original=magnetization[ir,2]
        magnetization[ir,2]=elem_magnetization
        ener2,ener_free2, entr2, onsager2,q2,q42=energy_entropy(interaction,magnetization,NN,Temp)
        Delta=ener2-ener
        num_rep.append(0)
        #print("Delta")
        #print(Delta)
        if(Delta<=0):
            ener=ener+Delta
            magnetization[ir,2]=np.sign(elem_magnetization)
            #print("entra a cambiar sin reparos")
        else:
            p=np.random.random()
            ee=np.exp(-b*Delta)
            if(p<ee):
                ener=ener+Delta
                magnetization[ir,2]=np.sign(elem_magnetization)
                #print("entro a cambiar con prob")
            else:
                magnetization[ir,2]=original
                ener=ener
                num_rep[i]+=1
                if num_rep[i-1]==1 and num_rep[i-2]==1 and num_rep[i-3]==1 and num_rep[i-4]==1 and num_rep[i-5]==1 and num_rep[i-6]==1 and num_rep[i-7]==1:
                    pos_rep.append(i)
                #print("entra a no cambiar, i=", i)
                        
                
                #print("no cambio")
        energ_list.append(float("{0:.10f}".format(ener/6)))
        t.append(i+1)
    return energ_list,t,num_rep,pos_rep

#multi=np.linspace(0.01,1.5,90)
#time=2000
#multi=[0.1]
#mat_interaction=acoplamientos(20)
#ener_int_list_list, q_list_list, q_average_list, total_fixed, energy_distr_list, q4, energy_free, real_av_list =muestreo(30,500,NN,multi,mat_interaction)
#print("ener_int_list_list=")
#print(ener_int_list_list)
#energiamonte,t=montecarlo(mat_interaction,NN,0.1,time)
#
#print("maximo de la energia tap=",max(ener_int_list_list[0]))
##print(energiamonte)
##print("lista de temperaturas de q, con longitud de=",len(q_list_list))
##print(q_list_list)
##print("pesos de cada q para cada temperatura, longitud=", len(energy_distr_list))
##print(energy_distr_list)
##print("fixed points")
##print(total_fixed)
#FIG=plt.figure()
#plt.plot(t,energiamonte, 'go--', linewidth=1, markersize=1, label = "Energy")
#plt.axhline(y=max(ener_int_list_list[0]))
#plt.xlabel("Time")
#plt.ylabel("E")
#plt.title("E(t)",fontsize=20)
#plt.legend()
#plt.grid()
#FIG.savefig("Energy(t),png")


def hist_ene(multi,NN,samples,num_mag):
#    print("entra al metodo con spines=",NN)
    try:
        os.mkdir('N={}'.format(NN))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    os.chdir('N={}'.format(NN))
    how_many_energies_list=[]
    how_many_energies=[]
    how_many_energies_list_free=[]
    how_may_q_list=[]
    multi2=multi
    for k in range(samples):
#        print("spines",NN)
        mat_interaction=acoplamientos(NN)
        ener_int_list_list, q_list_list, q_average_list, total_fixed, energy_distr_list, q4, energy_free, real_av_list =muestreo(num_mag,400,NN,multi2,mat_interaction)
#       aca etraria el otro for
        
        for i in range(len(multi2)):
#            print("temp=",multi[i])
            if k==0:
                how_many_energies_list.append([])
                how_many_energies_list_free.append([])
                how_may_q_list.append([])
#            print("energia antes")
#            print(how_many_energies_list)
#            print("energia free antes")
#            print(how_many_energies_list_free)                
            how_many_energies_list[i]+=ener_int_list_list[i]
            how_many_energies_list_free[i]+=energy_free[i]
            
#            print("energia despues")
#            print(how_many_energies_list)
#            print("energia free despues")
#            print(how_many_energies_list_free)
            
            how_may_q_list[i].append(real_av_list[i])
            if k==(samples-1):
                filter_filter=list(set(how_many_energies_list[i]))
                how_many_energies_per_t=len(filter_filter)
                how_many_energies.append(how_many_energies_per_t)
#        print("lista de q provenientes del metodo")
#        print(real_av_list)
#    print("how_many_energies_list")
#    print(how_many_energies_list)
#    print("how_many_energies_list_free")
#    print(how_many_energies_list_free)
#    print("how_many_energies")
#    print(how_many_energies)
#    print("how_many_q_list")
#    print(how_may_q_list)
#    print("number of soutions=", how_many_energies )
#    print("number of Temps=", len(how_many_energies_list))
#    print("ready to do histograms")
    for i in range(len(multi2)):
        try:
            os.mkdir('T={}'.format(multi2[i]))
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise    
#        os.mkdir('T={}'.format(multi[i]))
        os.chdir('T={}'.format(multi2[i]))
        HIS_en=plt.figure()
        n, bins, patches = plt.hist(x=how_many_energies_list[i],bins=(len(how_many_energies_list[i])*2),color='#0504aa')
        plt.xlabel('Interacion energy')
        plt.ylabel('Counts')
        plt.title('Interaction energy histogram')
        #plt.subplots_adjust(left=0.15)
        HIS_en.savefig("Interaction energy histogram_N={}_T={}.png".format(NN,multi2[i]))
        plt.close()
        
        HIS_q=plt.figure()
        n2, bins2, patches2 = plt.hist(x=how_may_q_list[i],bins=(len(how_may_q_list[i])*2),color='#0504aa')
        plt.xlabel('q_EA')
        plt.ylabel('Counts')
        plt.title('q_EA Histogram')
        #plt.subplots_adjust(left=0.15)
        HIS_q.savefig("Ed-And histogram_N={}_T={}.png".format(NN,multi2[i]))
        plt.close()
        
        HIS_free=plt.figure()
        n3, bins3, patches3 = plt.hist(x=how_many_energies_list_free[i],bins=(len(how_many_energies_list_free[i])*2),color='#0504aa')
        plt.xlabel('Free energy')
        plt.ylabel('Counts')
        plt.title('Free energy histogram')
        #plt.subplots_adjust(left=0.15)
        HIS_free.savefig("Free energy histogram_N={}_T={}.png".format(NN,multi2[i]))
        plt.close()
        os.chdir('../')
    file = open("number of solutions for N={}.txt".format(NN), "w")
    for n in range(len(how_many_energies)):
        file.write("{}    {}".format(multi2[n],how_many_energies[n]) + os.linesep)
    file.close()
#    print("acabo")
    os.chdir('../')
    
    
    
##################qEA dependence in T####################    
# NN=20
# num_mag=15
# multi=[0.1]
# samples=1
# try:
#     os.mkdir('q_Ea_N={}'.format(NN))
# except OSError as e:
#     if e.errno != errno.EEXIST:
#         raise
# os.chdir('q_Ea_N={}'.format(NN))
# how_many_energies_list=[]
# how_many_energies=[]
# how_many_energies_list_free=[]
# how_may_q_list=[]
# multi2=multi
# for k in range(samples):
#   print("spines",NN)
#    mat_interaction=acoplamientos(NN)
#    ener_int_list_list, q_list_list, q_average_list, total_fixed, energy_distr_list, q4, energy_free, real_av_list =muestreo(num_mag,400,NN,multi,mat_interaction)
#   aca etraria el otro for
#os.chdir('../')
###########################3 histogramas de la carpeta histogramas padres######################### recien comentado

#Temps=[0.1,0.15]
#Temps=[0.2,0.25]
#Temps=[0.3,0.35]
#Temps=[0.4,0.45]
#Temps=[0.5,0.55]
#numero=14
#mag_ini=55
#hist_ene(Temps,numero,150,mag_ini)
##    os.mkdir('prueba_{}'.format(i))
##    os.chdir('prueba_{}'.format(i))
#    file = open("filename_{}_{}.txt".format(i,'pr'), "w")
#    file.write("Primera línea" + os.linesep)
#    file.write("Segunda línea")
#    file.close()
#    os.chdir('../')
        
    
    
def ener_igua(ene_tap,ene_mon):
    print('------------energia montecarlo y tap, correspondientemente-----------')
    print(set(ene_mon),'------',ene_tap)
#    a=list(set(ene_mon).intersection(ene_tap))
    a=list(set(ene_mon) & set(ene_tap))
    return(a)
        
    
    
    
#############montecarlo################33    


te=0.3
init=time.time()

monte=np.load("4/enermonte.npy")
repetidos=np.load("4/pos-rep.npy")


print(monte[1754:1759])
print(monte[1805:1826])
monte=list(monte)
# print(monte.index(monte[1754], 1761, 5001))
monte2=[]
for elem in monte:
#    monte2.append("%.4f"%elem)
    monte2.append(elem)


t=range(5001)
# print(repetidos)
print(monte[1664],monte[1637])
if monte[1664]-monte[1637]==0:
    print("veo lok")
heh=[]
for elem in repetidos:
    heh2=[]
    for i in range(len(monte2)):
        if monte2[i]==monte2[elem]:
            heh2.append(i)
        else:
            None 
    heh.append(heh2)

# with open("holi.csv", "a", newline='') as f:
#     wr = csv.writer(f)
#     wr.writerows(heh)




# time=5000
# NN=20
# numero=0
# mat_interaction=np.load("matriz_J.npy")
# magnetizaciones=np.load("list_qea_inter_free_N=20_array_{}.npy".format(numero))

# #print(magnetizaciones[1])
# numero2=random.randint(0, len(magnetizaciones)-1)
# print(numero2)
# energiamonte,t, num_rep,pos_rep=montecarlo(mat_interaction,NN,te,time)
# print(ener_igua(entap,energiamonte))
# #ener_int_list_list, q_list_list, q_average_list, total_fixed, energy_distr_list, q4, energy_free, real_av_list =muestreo(40,400,NN,multi,mat_interaction)
# #tap_and_mont=ener_igua(ener_int_list_list[0],energiamonte)
# #print("tap y monte")
# #print(tap_and_mont)
# #print("----------------lista de energias de interaccion-------------")
# #print(ener_int_list_list)
# #print("-------------lista de q-----------------")
# #print(q_list_list)
# #print("----------------average crudo------------------")
# #print(q_list_list)
# #print("--------------------average de verdad------------------")
# #print(real_av_list)




# FIG=plt.figure()
# plt.plot(t,monte, 'go-', linewidth=0.5, markersize=1, label = "MC")
# #plt.axhline(y=max(ener_int_list_list[20]))
# # plt.axhline(y=energiamonte[pos_rep[int(len(pos_rep)/2)]], label="TAP")
# # plt.axhline(y=energiamonte[pos_rep[int(len(pos_rep)/3)]], label="TAP")
# plt.xlim(0,len(t))
# #plt.ylim(energiamonte[pos_rep[int(len(pos_rep)/2)]]-0.1,energiamonte[pos_rep[int(len(pos_rep)/3)]]+0.1)
# plt.xlabel("t")
# plt.ylabel("E")
# #plt.title("E(t)",fontsize=20)
# plt.legend()
# plt.grid()
# FIG.savefig('Energy(t).png')



fig, (ax1,ax2) = plt.subplots(1, 2,sharey='row')

ax1.plot(t[200:600], monte[200:600],'go-', linewidth=0.7, markersize=1.2, label = "MC")
ax1.axhline(y=monte[490], linewidth=0.5,label="TAP")
ax1.axhline(y=monte[940], linewidth=0.5, label="TAP")
ax2.plot(t[800:1000], monte[800:1000],'go-', linewidth=0.5, markersize=1, label = "MC")
plt.ylim(-0.6,-0.45)
ax2.axhline(y=monte[490], linewidth=0.5, label="TAP")
ax2.axhline(y=monte[940], linewidth=0.5, label="TAP")
# ax1.xlabel('t')
# ax2.xlabel('t')
ax1.set(xlabel='t',ylabel='e(t)')
ax2.set(xlabel='t')
plt.legend()
plt.savefig('nueva.png')


# FIG2=plt.figure()
# plt.plot(t,monte, 'go-', linewidth=0.5, markersize=1, label = "MC")
# #plt.axhline(y=max(ener_int_list_list[20]))
# plt.axhline(y=monte[1754], label="TAP")
# plt.axhline(y=monte[1826], label="TAP")
# plt.xlim(1500,2500)
# plt.ylim(-0.6,-0.45)
# plt.xlabel("t")
# plt.ylabel("E")
# #plt.title("E(t)",fontsize=20)
# plt.legend()
# plt.grid()
# FIG2.savefig('Energy(t)_2.png')



    
def dif_T(samples,number_mag,ticks,N,T):
    q_each_T_plane_av=[]
    cont=0
    for i in T:
        cont+=1
        print(i,cont)
        temp=i
        beta = 1./temp
        ener_int_list_list,q_list_list, q_average_list, total_fixed,energy_distr_list,q4,energy_free =muestreo(samples,number_mag,ticks,NN,beta)
        ener_int_list_list, q_list_list, q_average_list, total_fixed, energy_distr_list, q4, energy_free, real_av_list =muestreo(30,500,NN,multi,mat_interaction)
        print("total fixed")
        print(total_fixed)
        print(energy_distr_list)
        print("lista de los q")
        print(q_list_list)
        a=list(set(q_list_list))
        q_av=np.average(q_average_list)
        q_each_T_plane_av.append(q_av)
    return q_each_T_plane_av


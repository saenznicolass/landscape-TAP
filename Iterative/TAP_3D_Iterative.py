
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
        #print(energy,number_fixed_single_mag)
        #print("q=",q)
        #print("punto encontrado")
    ener_int=ener_int/6
    magnet=mag
    return ener_int,energy_free, number_fixed_single_mag, q,q4,magnet, fixed


def over_magnetizations(number_magnetizations,ticks,interaction,N,beta):
    number_fixed_J=0
    energytot_free=[]
    q_tot=[]
    q4_tot=[]
    ener_interaction=[]
    magneti=[1]
    for i in range (number_magnetizations):
        #print("-----numero de matriz de magnetizacion:",i,"--------")
        ener_int,energy_free,number_fixed_single_mag,q,q4,magter,fixed=over_time(ticks,interaction,N,beta,magneti,0)
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
                if num_rep[i-1]==1 and num_rep[i-2]==1:
                    pos_rep.append(i)
                #print("entra a no cambiar, i=", i)
                        
                
                #print("no cambio")
        energ_list.append(ener/6)
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
        plt.xlim(-1.5,0)
        #plt.subplots_adjust(left=0.15)
        HIS_en.savefig("Interaction energy histogram_N={}_T={}.png".format(NN,multi2[i]))
        plt.close()
        
        HIS_q=plt.figure()
        n2, bins2, patches2 = plt.hist(x=how_may_q_list[i],bins=(len(how_may_q_list[i])*2),color='#0504aa')
        plt.xlabel('q_EA')
        plt.ylabel('Counts')
        plt.title('q_EA Histogram')
        plt.xlim(0,1)
        #plt.subplots_adjust(left=0.15)
        HIS_q.savefig("Ed-And histogram_N={}_T={}.png".format(NN,multi2[i]))
        plt.close()
        
        HIS_free=plt.figure()
        n3, bins3, patches3 = plt.hist(x=how_many_energies_list_free[i],bins=(len(how_many_energies_list_free[i])*2),color='#0504aa')
        plt.xlabel('Free energy')
        plt.ylabel('Counts')
        plt.title('Free energy histogram')
        plt.xlim(-2,0)
        #plt.subplots_adjust(left=0.15)
        HIS_free.savefig("Free energy histogram_N={}_T={}.png".format(NN,multi2[i]))
        plt.close()
        
        file = open("Interaction energies for N={}_T={}.txt".format(NN,multi2[i]), "w+")
        for n in range(len(q_average_list)):
            file.write(how_many_energies_list[i] + os.linesep)
        file.close()
        
        file = open("Free energies for N={}_T={}.txt".format(NN,multi2[i]), "w+")
        for n in range(len(q_average_list)):
            file.write(how_may_q_list[i] + os.linesep)
        file.close()
        
        file = open("EA parameters for N={}_T={}.txt".format(NN,multi2[i]), "w+")
        for n in range(len(q_average_list)):
            file.write(how_many_energies_list[i] + os.linesep)
        file.close()
        
        os.chdir('../')
    file = open("number of solutions for N={}.txt".format(NN), "w")
    for n in range(len(how_many_energies)):
        file.write("{}    {}".format(multi2[n],how_many_energies[n]) + os.linesep)
    file.close()
#    print("acabo")
    os.chdir('../')
    
###########################3 histogramas de la carpeta histogramas padres######################### recien comentado

# Temps=[0.1]
# numero=20
# mag_ini=20
# hist_ene(Temps,numero,5,mag_ini)
##    os.mkdir('prueba_{}'.format(i))
##    os.chdir('prueba_{}'.format(i))
#    file = open("filename_{}_{}.txt".format(i,'pr'), "w")
#    file.write("Primera línea" + os.linesep)
#    file.write("Segunda línea")
#    file.close()
#    os.chdir('../')
        
    
    
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
            try:
                hess=hessian(interaction,magne[:,2],N,beta)
            except:
                continue
            finally:
                if hess is not None:
                    
            # print("-----------hessian--------------")
            # print(hess)
                    eigen=np.linalg.eigvals(hess)
                    hessian_eig.append(eigen)
                else:
                    None
            # print("-------------eigen-----------")
            # print(eigen)
#    print("last T=", te)        
    return te, energias, hessian_eig,list_t_lim, list_q
    
multi=np.linspace(0.15,0.8,60)
interaction=acoplamientos(20)
# for i in range(20):

#     te,en,eigen=initial_iterative(100, interaction, 20, multi)
#     # FIG=plt.figure()
#     # plt.plot(t,energiamonte, 'go--', linewidth=1, markersize=1, label = "Energy")
#     # plt.xlabel("Time")
#     # plt.ylabel("E")
#     # plt.title("E(t)",fontsize=20)
#     # plt.legend()
#     # plt.grid()
#     # FIG.savefig('Energy(t).png')
  

list_list_ene=[]
list_list_temp=[]
list_q_general=[]

for i in range(6000):
    te=None
    en=[]
    eigen=None
    list_t=[]
    list_q=[]
    try:
        te,en,eigen,list_t,list_q=initial_iterative(200, interaction, 20, multi)
    except: 
        continue
    finally:
        if len(en)==0 :
            pass
        else:
            # print("---lista de t-----")
            # print(list_t)
            # print("----energias")
            # print(en)
            list_list_ene.append(en)
            list_list_temp.append(list_t)
            list_q_general.append(list_q)
            FIG=plt.figure(figsize=(17,10))
            plt.plot(list_t,en, label='TAP')
            plt.xlabel("T", size=33)
            plt.ylabel("E", size=33)
            plt.yticks(fontsize=26)
            plt.xticks(fontsize=26)
            plt.legend()
            FIG.savefig('energy-vs-T-{}-3.png'.format(i))


np.save("list_list_temp.npy",list_list_temp)
np.save("list_list_ene.npy",list_list_ene)
np.save("list_q_general.npy",list_q_general)

FIG2=plt.figure(figsize=(17,10))
plt.xlabel("T",size=33)
plt.ylabel("e",size=33)
for i in range(len(list_list_ene)):
    plt.plot(list_list_temp[i],list_list_ene[i])
    plt.yticks(fontsize=26)
    plt.xticks(fontsize=26)
#plt.legend()
FIG2.savefig('energy-vs-T-master-3.png')
    
q_a_plot = [item for sublist in list_q_general for item in sublist]
t_a_plot_in_q = [item for sublist in list_list_temp for item in sublist]
e_a_plot_in_q = [item for sublist in list_list_ene for item in sublist]
FIG3=plt.figure(figsize=(17,10))
plt.xlabel("T", size=33)
plt.ylabel("$q_{EA}$", size=33)
plt.scatter(t_a_plot_in_q,q_a_plot,s=10)
plt.yticks(fontsize=26)
plt.xticks(fontsize=26)
FIG3.savefig('q-EA-vs-T-3.png')

FIG4=plt.figure(figsize=(17,10))
plt.xlabel("E", size=33)
plt.ylabel("$q_{EA}$", size=33)
plt.scatter(e_a_plot_in_q,q_a_plot,s=10)
plt.yticks(fontsize=26)
plt.xticks(fontsize=26)
FIG4.savefig('q-EA-vs-energy-3.png')

    


#############montecarlo################33    
#mat_interaction=acoplamientos(20)
#time=1000
#NN=20
#multi=[0.3]
#energiamonte,t, num_rep,pos_rep=montecarlo(mat_interaction,NN,0.1,time)
##ener_int_list_list, q_list_list, q_average_list, total_fixed, energy_distr_list, q4, energy_free, real_av_list =muestreo(40,400,NN,multi,mat_interaction)
##tap_and_mont=ener_igua(ener_int_list_list[0],energiamonte)
##print("tap y monte")
##print(tap_and_mont)
##print("----------------lista de energias de interaccion-------------")
##print(ener_int_list_list)
##print("-------------lista de q-----------------")
##print(q_list_list)
##print("----------------average crudo------------------")
##print(q_list_list)
##print("--------------------average de verdad------------------")
##print(real_av_list)
#FIG=plt.figure()
#plt.plot(t,energiamonte, 'go--', linewidth=1, markersize=1, label = "Energy")
##plt.axhline(y=max(ener_int_list_list[20]))
#plt.axhline(y=energiamonte[pos_rep[len(pos_rep)-9]], label="TAP Energy={}".format(energiamonte[pos_rep[len(pos_rep)-9]]))
#plt.axhline(y=energiamonte[pos_rep[len(pos_rep)-18]], label="TAP Energy={}".format(energiamonte[pos_rep[len(pos_rep)-18]]))
#plt.xlim(600,999)
#plt.ylim(energiamonte[pos_rep[len(pos_rep)-9]]-0.1,energiamonte[pos_rep[len(pos_rep)-18]]+0.1)
#plt.xlabel("Time")
#plt.ylabel("E")
#plt.title("E(t)",fontsize=20)
#plt.legend()
#plt.grid()
#FIG.savefig('Energy(t).png')
#    
#def dif_T(samples,number_mag,ticks,N,T):
#    q_each_T_plane_av=[]
#    cont=0
#    for i in T:
#        cont+=1
#        print(i,cont)
#        temp=i
#        beta = 1./temp
#        ener_int_list_list,q_list_list, q_average_list, total_fixed,energy_distr_list,q4,energy_free =muestreo(samples,number_mag,ticks,NN,beta)
#        ener_int_list_list, q_list_list, q_average_list, total_fixed, energy_distr_list, q4, energy_free, real_av_list =muestreo(30,500,NN,multi,mat_interaction)
#        print("total fixed")
#        print(total_fixed)
#        print(energy_distr_list)
#        print("lista de los q")
#        print(q_list_list)
#        a=list(set(q_list_list))
#        q_av=np.average(q_average_list)
#        q_each_T_plane_av.append(q_av)
#    return q_each_T_plane_av


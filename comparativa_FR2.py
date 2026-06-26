#%% Librerias y paquetes 
import numpy as np
from uncertainties import ufloat, unumpy
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import os
import chardet
import re
from clase_resultados import ResultadosESAR
pendiente_HvsI = 3716.3 # 1/m
ordenada_HvsI = 1297.0 # A/m
#%% Lector de resultados
def lector_resultados(path):
    '''
    Para levantar archivos de resultados con columnas :
    Nombre_archivo	Time_m	Temperatura_(ºC)	Mr_(A/m)	Hc_(kA/m)	Campo_max_(A/m)	Mag_max_(A/m)	f0	mag0	dphi0	SAR_(W/g)	Tau_(s)	N	xi_M_0
    '''
    with open(path, 'rb') as f:
        codificacion = chardet.detect(f.read())['encoding']

    # Leer las AUV2s 20 líneas y crear un diccionario de meta
    meta = {}
    with open(path, 'r', encoding=codificacion) as f:
        for i in range(20):
            line = f.readline()
            if i == 0:
                match = re.search(r'Rango_Temperaturas_=_([-+]?\d+\.\d+)_([-+]?\d+\.\d+)', line)
                if match:
                    key = 'Rango_Temperaturas'
                    value = [float(match.group(1)), float(match.group(2))]
                    meta[key] = value
            else:
                # Patrón para valores con incertidumbre (ej: 331.45+/-6.20 o (9.74+/-0.23)e+01)
                match_uncertain = re.search(r'(.+)_=_\(?([-+]?\d+\.\d+)\+/-([-+]?\d+\.\d+)\)?(?:e([+-]\d+))?', line)
                if match_uncertain:
                    key = match_uncertain.group(1)[2:]  # Eliminar '# ' al inicio
                    value = float(match_uncertain.group(2))
                    uncertainty = float(match_uncertain.group(3))
                    
                    # Manejar notación científica si está presente
                    if match_uncertain.group(4):
                        exponent = float(match_uncertain.group(4))
                        factor = 10**exponent
                        value *= factor
                        uncertainty *= factor
                    
                    meta[key] = ufloat(value, uncertainty)
                else:
                    # Patrón para valores simples (sin incertidumbre)
                    match_simple = re.search(r'(.+)_=_([-+]?\d+\.\d+)', line)
                    if match_simple:
                        key = match_simple.group(1)[2:]
                        value = float(match_simple.group(2))
                        meta[key] = value
                    else:
                        # Capturar los casos con nombres de archivo
                        match_files = re.search(r'(.+)_=_([a-zA-Z0-9._]+\.txt)', line)
                        if match_files:
                            key = match_files.group(1)[2:]
                            value = match_files.group(2)
                            meta[key] = value

    # Leer los datos del archivo (esta parte permanece igual)
    data = pd.read_table(path, header=15,
                         names=('name', 'Time_m', 'Temperatura',
                                'Remanencia', 'Coercitividad','Campo_max','Mag_max',
                                'frec_fund','mag_fund','dphi_fem',
                                'SAR','tau',
                                'N','xi_M_0'),
                         usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13),
                         decimal='.',
                         engine='python',
                         encoding=codificacion)

    files = pd.Series(data['name'][:]).to_numpy(dtype=str)
    time = pd.Series(data['Time_m'][:]).to_numpy(dtype=float)
    temperatura = pd.Series(data['Temperatura'][:]).to_numpy(dtype=float)
    Mr = pd.Series(data['Remanencia'][:]).to_numpy(dtype=float)
    Hc = pd.Series(data['Coercitividad'][:]).to_numpy(dtype=float)
    campo_max = pd.Series(data['Campo_max'][:]).to_numpy(dtype=float)
    mag_max = pd.Series(data['Mag_max'][:]).to_numpy(dtype=float)
    xi_M_0=  pd.Series(data['xi_M_0'][:]).to_numpy(dtype=float)
    SAR = pd.Series(data['SAR'][:]).to_numpy(dtype=float)
    tau = pd.Series(data['tau'][:]).to_numpy(dtype=float)

    frecuencia_fund = pd.Series(data['frec_fund'][:]).to_numpy(dtype=float)
    dphi_fem = pd.Series(data['dphi_fem'][:]).to_numpy(dtype=float)
    magnitud_fund = pd.Series(data['mag_fund'][:]).to_numpy(dtype=float)

    N=pd.Series(data['N'][:]).to_numpy(dtype=int)
    return meta, files, time,temperatura,Mr, Hc, campo_max, mag_max, xi_M_0, frecuencia_fund, magnitud_fund , dphi_fem, SAR, tau, N
#%% LECTOR CICLOS
def lector_ciclos(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()[:8]

    metadata = {'filename': os.path.split(filepath)[-1],
                'Temperatura':float(lines[0].strip().split('_=_')[1]),
        "Concentracion_g/m^3": float(lines[1].strip().split('_=_')[1].split(' ')[0]),
            "C_Vs_to_Am_M": float(lines[2].strip().split('_=_')[1].split(' ')[0]),
            "pendiente_HvsI": float(lines[3].strip().split('_=_')[1].split(' ')[0]),
            "ordenada_HvsI": float(lines[4].strip().split('_=_')[1].split(' ')[0]),
            'frecuencia':float(lines[5].strip().split('_=_')[1].split(' ')[0])}

    data = pd.read_table(os.path.join(os.getcwd(),filepath),header=7,
                        names=('Tiempo_(s)','Campo_(Vs)','Magnetizacion_(Vs)','Campo_(kA/m)','Magnetizacion_(A/m)'),
                        usecols=(0,1,2,3,4),
                        decimal='.',engine='python',
                        dtype= {'Tiempo_(s)':'float','Campo_(Vs)':'float','Magnetizacion_(Vs)':'float',
                               'Campo_(kA/m)':'float','Magnetizacion_(A/m)':'float'})
    t     = pd.Series(data['Tiempo_(s)']).to_numpy()
    H_Vs  = pd.Series(data['Campo_(Vs)']).to_numpy(dtype=float) #Vs
    M_Vs  = pd.Series(data['Magnetizacion_(Vs)']).to_numpy(dtype=float)#A/m
    H_kAm = pd.Series(data['Campo_(kA/m)']).to_numpy(dtype=float)*1000 #A/m
    M_Am  = pd.Series(data['Magnetizacion_(A/m)']).to_numpy(dtype=float)#A/m

    return t,H_Vs,M_Vs,H_kAm,M_Am,metadata
#%% funcion extraer SAR, tau y Hc de resultados 
def extraer_SAR_tau(resultados):
    SAR = []
    tau = []
    Hc = []
    for res in resultados:
        meta,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = lector_resultados(res)   
        SAR.append(meta['SAR_W/g'])
        tau.append(meta['tau_ns'])
        Hc.append(meta['Hc_kA/m']) 
    return np.array(SAR), np.array(tau), np.array(Hc)
#%% Ciclos y resultados para 300, 268 , 213 135 y 081 
conc_FR =  20 #g/L
# 300
ciclos_FR_300 = glob("300/**/*ciclo_promedio_H_M.txt",recursive=True)
ciclos_FR_300.sort(reverse=True)
resultados_FR_300 = glob("300/**/*resultados.txt",recursive=True)
resultados_FR_300.sort(reverse=True)

for p in ciclos_FR_300:
    print('  ',os.path.split(p)[-1])
print('-'*50)    

for p in resultados_FR_300:
    print('  ',os.path.split(p)[-1])
print('-'*50)    

#268
ciclos_FR_268 = glob("268/**/*ciclo_promedio_H_M.txt",recursive=True)
ciclos_FR_268.sort()
resultados_FR_268 = glob("268/**/*resultados.txt",recursive=True)
resultados_FR_268.sort()

for p in ciclos_FR_268:
    print('  ',os.path.split(p)[-1])
print('-'*50)    

for p in resultados_FR_268:
    print('  ',os.path.split(p)[-1])
print('-'*50)    

# 213
ciclos_FR_213 = glob("213/**/*ciclo_promedio_H_M.txt",recursive=True)
ciclos_FR_213.sort()
resultados_FR_213 = glob("213/**/*resultados.txt",recursive=True)
resultados_FR_213.sort()

for p in ciclos_FR_213:
    print('  ',os.path.split(p)[-1])
print('-'*50)    

for p in resultados_FR_213:
    print('  ',os.path.split(p)[-1])
print('-'*50)    
# 135
ciclos_FR_135 = glob("135/**/*ciclo_promedio_H_M.txt",recursive=True)
ciclos_FR_135.sort()
resultados_FR_135 = glob("135/**/*resultados.txt",recursive=True)
resultados_FR_135.sort()

for p in ciclos_FR_135:
    print('  ',os.path.split(p)[-1])
print('-'*50)    

for p in resultados_FR_135:
    print('  ',os.path.split(p)[-1])
print('-'*50)    
#081
ciclos_FR_081 = glob("081/**/*ciclo_promedio_H_M.txt",recursive=True)
ciclos_FR_081.sort()
resultados_FR_081 = glob("081/**/*resultados.txt",recursive=True)
resultados_FR_081.sort()

for p in ciclos_FR_081:
    print('  ',os.path.split(p)[-1])
print('-'*50)    

for p in resultados_FR_081:
    print('  ',os.path.split(p)[-1])
print('-'*50)    

#%% ploteo ciclos 
idc=[15,10,8,6,4,1.5]
H0 = [(h*pendiente_HvsI+ordenada_HvsI)/1000 for h in idc]

fig10, ax =plt.subplots(figsize=(10,7.5),constrained_layout=True,sharey=True,sharex=True)
for i,e in enumerate(ciclos_FR_300):
    _,_,_, H_FR,M_FR,_ = lector_ciclos(ciclos_FR_300[i])
    ax.plot(H_FR/1000,M_FR,'-',label=f'{H0[i]:.1f}')
    
ax.set_xticks([-round(h,1) for h in H0]+ [0] + [round(h2,1) for h2 in H0[::-1]],rotation=45)
ax.tick_params(axis='x', labelrotation=45)
ax.grid()
ax.set_ylabel('M (A/m)')
ax.set_xlabel('H (kA/m)')
ax.legend(loc='upper left',frameon=True,shadow=True,title='H$_0$ (kA/m)',ncol=2)
plt.suptitle(f'Ciclos promedio FR \n300 kHz  [57; 6.9] kA/m')
plt.show()

fig00, ax =plt.subplots(figsize=(10,7.5),constrained_layout=True,sharey=True,sharex=True)
for i,e in enumerate(ciclos_FR_268):
    _,_,_, H_FR,M_FR,_ = lector_ciclos(ciclos_FR_268[i])
    ax.plot(H_FR/1000,M_FR,'-',label=f'{H0[i]:.1f}')
    
ax.set_xticks([-round(h,1) for h in H0]+ [0] + [round(h2,1) for h2 in H0[::-1]],rotation=45)
ax.tick_params(axis='x', labelrotation=45)
ax.grid()
ax.set_ylabel('M (A/m)')
ax.set_xlabel('H (kA/m)')
ax.legend(loc='upper left',frameon=True,shadow=True,title='H$_0$ (kA/m)',ncol=2)
plt.suptitle(f'Ciclos promedio FR \n268 kHz  [57; 6.9] kA/m')
plt.show()

fig01, ax =plt.subplots(figsize=(10,7.5),constrained_layout=True,sharey=True,sharex=True)
for i,e in enumerate(ciclos_FR_213):
    _,_,_, H_FR,M_FR,_ = lector_ciclos(ciclos_FR_213[i])
    ax.plot(H_FR/1000,M_FR,'-',label=f'{H0[i]:.1f}')
ax.set_xticks([-round(h,1) for h in H0]+ [0] + [round(h2,1) for h2 in H0[::-1]],rotation=45)
ax.tick_params(axis='x', labelrotation=45)
ax.grid()
ax.set_ylabel('M (A/m)')
ax.set_xlabel('H (kA/m)')
ax.legend(loc='upper left',frameon=True,shadow=True,title='H$_0$ (kA/m)',ncol=2)
plt.suptitle(f'Ciclos promedio FR \n213 kHz  [57; 6.9] kA/m')
plt.show()

fig02, ax =plt.subplots(figsize=(10,7.5),constrained_layout=True,sharey=True,sharex=True)
for i,e in enumerate(ciclos_FR_135):
    _,_,_, H_FR,M_FR,_ = lector_ciclos(ciclos_FR_135[i])
    ax.plot(H_FR/1000,M_FR,'-',label=f'{H0[i]:.1f}')
ax.set_xticks([-round(h,1) for h in H0]+ [0] + [round(h2,1) for h2 in H0[::-1]],rotation=45)
ax.tick_params(axis='x', labelrotation=45)
ax.grid()
ax.set_ylabel('M (A/m)')
ax.set_xlabel('H (kA/m)')
ax.legend(loc='upper left',frameon=True,shadow=True,title='H$_0$ (kA/m)',ncol=2)
plt.suptitle(f'Ciclos promedio FR \n135 kHz  [57; 6.9] kA/m')
plt.show()


fig03, ax =plt.subplots(figsize=(10,7.5),constrained_layout=True,sharey=True,sharex=True)
for i,e in enumerate(ciclos_FR_081):
    _,_,_, H_FR,M_FR,_ = lector_ciclos(ciclos_FR_081[i])
    ax.plot(H_FR/1000,M_FR,'-',label=f'{H0[i]:.1f}')
ax.set_xticks([-round(h,1) for h in H0]+ [0] + [round(h2,1) for h2 in H0[::-1]],rotation=45)
ax.tick_params(axis='x', labelrotation=45)
ax.grid()
ax.set_ylabel('M (A/m)')
ax.set_xlabel('H (kA/m)')
ax.legend(loc='upper left',frameon=True,shadow=True,title='H$_0$ (kA/m)',ncol=2)
plt.suptitle(f'Ciclos promedio FR \n081 kHz  [57; 6.9] kA/m')
plt.show()

#%% comparo las 5 medidas

fig04, axs = plt.subplots(2,3,figsize=(14,8),constrained_layout=True,sharex=True,sharey=True)

axs[0,0].set_title('300 kHz',loc='left')
axs[0,1].set_title('268 kHz',loc='left')
axs[0,2].set_title('213 kHz',loc='left')
axs[1,0].set_title('135 kHz',loc='left')
axs[1,1].set_title('081 kHz',loc='left')

axs[1,2].axis('off')


for i,e in enumerate(ciclos_FR_300):
    _,_,_, H_FR,M_FR,_ = lector_ciclos(e)
    axs[0,0].plot(H_FR/1000,M_FR,label=f'{H0[i]:.1f}')

for i,j in enumerate(ciclos_FR_268):
    _,_,_, H_FR,M_FR,_ = lector_ciclos(j)
    axs[0,1].plot(H_FR/1000,M_FR,label=f'{H0[i]:.1f}')

for i,k in enumerate(ciclos_FR_213):
    _,_,_, H_FR,M_FR,_ = lector_ciclos(k)
    axs[0,2].plot(H_FR/1000,M_FR,label=f'{H0[i]:.1f}')

for i,l in enumerate(ciclos_FR_135):
    _,_,_, H_FR,M_FR,_ = lector_ciclos(l)
    axs[1,0].plot(H_FR/1000,M_FR,label=f'{H0[i]:.1f}')
    
for i,l in enumerate(ciclos_FR_081):
    _,_,_, H_FR,M_FR,_ = lector_ciclos(l)
    axs[1,1].plot(H_FR/1000,M_FR,label=f'{H0[i]:.1f}')
    
    
for a in axs.flatten()[:-1]:
    a.grid()
    a.legend(loc='upper left',frameon=True,shadow=True,title='H$_0$ (kA/m)',ncol=2)

for a in axs.flatten()[2:-1]:
    a.set_xlabel('H (kA/m)')

for a in [axs[0,0],axs[1,0]]:
    a.set_ylabel('M (A/m)')


plt.suptitle(f'Ciclos promedio FR  [57; 6.9] kA/m',fontsize=15)
# %% Comparo a mismo campo
f=[300,268,213,135,81]

ciclos_57=[ciclos_FR_300[0],ciclos_FR_268[0],ciclos_FR_213[0],ciclos_FR_135[0],ciclos_FR_081[0]] 
fig05,ax=plt.subplots(constrained_layout=True)
for i,e in enumerate(ciclos_57):
    _,_,_, H_FR,M_FR,_ = lector_ciclos(e)
    ax.plot(H_FR/1000,M_FR,'-',label=f'{f[i]}')
ax.grid()
ax.set_ylabel('M (A/m)')
ax.set_xlabel('H (kA/m)')
ax.legend(loc='upper left',frameon=True,shadow=True,title='f (kHz)',ncol=1)
plt.suptitle(f'Ciclos promedio FR \n57 kA/m')
plt.show()


ciclos_38=[ciclos_FR_300[1],ciclos_FR_268[1],ciclos_FR_213[1],ciclos_FR_135[1],ciclos_FR_081[1]] 
fig06,ax=plt.subplots(constrained_layout=True)
for i,e in enumerate(ciclos_38):
    _,_,_, H_FR,M_FR,_ = lector_ciclos(e)
    ax.plot(H_FR/1000,M_FR,'-',label=f'{f[i]}')
ax.grid()
ax.set_ylabel('M (A/m)')
ax.set_xlabel('H (kA/m)')
ax.legend(loc='upper left',frameon=True,shadow=True,title='f (kHz)',ncol=1)
plt.suptitle(f'Ciclos promedio FR \n 38.5 kA/m')
plt.show()

ciclos_31=[ciclos_FR_300[2],ciclos_FR_268[2],ciclos_FR_213[2],ciclos_FR_135[2],ciclos_FR_081[2]]
fig07,ax=plt.subplots(constrained_layout=True)
for i,e in enumerate(ciclos_31):
    _,_,_, H_FR,M_FR,_ = lector_ciclos(e)
    ax.plot(H_FR/1000,M_FR,'-',label=f'{f[i]}')
ax.grid()
ax.set_ylabel('M (A/m)')
ax.set_xlabel('H (kA/m)')
ax.legend(loc='upper left',frameon=True,shadow=True,title='f (kHz)',ncol=1)
plt.suptitle(f'Ciclos promedio FR \n 31.5 kA/m')
plt.show()
 
ciclos_23=[ciclos_FR_300[3],ciclos_FR_268[3],ciclos_FR_213[3],ciclos_FR_135[3],ciclos_FR_081[3]]
fig08,ax=plt.subplots(constrained_layout=True)
for i,e in enumerate(ciclos_23):
    _,_,_, H_FR,M_FR,_ = lector_ciclos(e)
    ax.plot(H_FR/1000,M_FR,'-',label=f'{f[i]}')
ax.grid()
ax.set_ylabel('M (A/m)')
ax.set_xlabel('H (kA/m)')
ax.legend(loc='upper left',frameon=True,shadow=True,title='f (kHz)',ncol=1)
plt.suptitle(f'Ciclos promedio FR \n 23 kA/m')
plt.show()

ciclos_16=[ciclos_FR_300[4],ciclos_FR_268[4],ciclos_FR_213[4],ciclos_FR_135[4],ciclos_FR_081[4]]
fig09,ax=plt.subplots(constrained_layout=True)
for i,e in enumerate(ciclos_16):
    _,_,_, H_FR,M_FR,_ = lector_ciclos(e)
    ax.plot(H_FR/1000,M_FR,'-',label=f'{f[i]}')
ax.grid()
ax.set_ylabel('M (A/m)')
ax.set_xlabel('H (kA/m)')
ax.legend(loc='upper left',frameon=True,shadow=True,title='f (kHz)',ncol=1)
plt.suptitle(f'Ciclos promedio FR \n 16 kA/m')
plt.show()

ciclos_7=[ciclos_FR_300[5],ciclos_FR_268[5],ciclos_FR_213[5],ciclos_FR_135[5],ciclos_FR_081[5]]
fig10,ax=plt.subplots(constrained_layout=True)
for i,e in enumerate(ciclos_7):
    _,_,_, H_FR,M_FR,_ = lector_ciclos(e)
    ax.plot(H_FR/1000,M_FR,'-',label=f'{f[i]}')
ax.grid()
ax.set_ylabel('M (A/m)')
ax.set_xlabel('H (kA/m)')
ax.legend(loc='upper left',frameon=True,shadow=True,title='f (kHz)',ncol=1)
plt.suptitle(f'Ciclos promedio FR \n 7 kA/m')
plt.show()
#%% 300 kHz a Bajo campo
# 300
ciclos_300_LF = glob("H_07_to_16/**/*ciclo_promedio_H_M.txt",recursive=True)
ciclos_300_LF.sort(reverse=True)
resultados_300_LF = glob("H_07_to_16/**/*resultados.txt",recursive=True)
resultados_300_LF.sort(reverse=True)

for p in ciclos_300_LF:
    print('  ',os.path.split(p)[-1])
print('-'*50)    

for p in resultados_300_LF:
    print('  ',os.path.split(p)[-1])
print('-'*50)    


idc=[4,3.5,3,2.5,2,1.5]
H0 = [(h*pendiente_HvsI+ordenada_HvsI)/1000 for h in idc]
fig11, ax = plt.subplots(figsize=(10,7.5),constrained_layout=True)

for i,e in enumerate(ciclos_300_LF):
    _,_,_, H_FR,M_FR,_ = lector_ciclos(e)
    ax.plot(H_FR/1000,M_FR,label=f'{H0[i]:.1f}')


ax.set_xticks([-round(h,1) for h in H0]+ [0] + [round(h2,1) for h2 in H0[::-1]],rotation=45)
ax.tick_params(axis='x', labelrotation=45)
ax.grid()
ax.set_ylabel('M (A/m)')
ax.set_xlabel('H (kA/m)')
ax.legend(loc='upper left',frameon=True,shadow=True,title='H$_0$ (kA/m)',ncol=2)
plt.suptitle(f'Ciclos FR \n300 kHz  [16; 6.9] kA/m')
plt.show()

#%% Salvo todas las figuras de ciclos
fig10.savefig('00_300_ciclos_prom.png',dpi=300)
fig00.savefig('00_268_ciclos_prom.png',dpi=300)
fig01.savefig('00_213_ciclos_prom.png',dpi=300)
fig02.savefig('00_135_ciclos_prom.png',dpi=300)
fig03.savefig('00_081_ciclos_prom.png',dpi=300)
#%%
fig04.savefig('01_ciclos_comparativa.png',dpi=300)
#%%
fig05.savefig('02_57_ciclos_prom.png',dpi=300)
fig06.savefig('02_38_ciclos_prom.png',dpi=300)
fig07.savefig('02_31_ciclos_prom.png',dpi=300)
fig08.savefig('02_23_ciclos_prom.png',dpi=300)
fig09.savefig('02_16_ciclos_prom.png',dpi=300)
fig10.savefig('02_07_ciclos_prom.png',dpi=300)
#%%
fig11.savefig('03_300_ciclos_LF.png',dpi=300)
#%% Ahora los resultados

#para 300 kHz incorporolas meedidas con idc entre 4 y 1.4
# resultados_FR_300_LF = glob("H_07_to_16/**/*resultados.txt",recursive=True)
# resultados_FR_300_LF.sort(reverse=True)

resultados_FR_300_all = glob("../260612_Pmag_FR/FRes_300kHz/**/*resultados.txt",recursive=True)
resultados_FR_300_all.sort(reverse=True)
#resultados_FR_300[:-2]+resultados_300_LF
SAR_300, tau_300, Hc_300 = extraer_SAR_tau(resultados_FR_300_all)
res_300=[]
print('Resultados 300 kHz', '='*80,'\n')
for r in resultados_FR_300:
    res_300.append(ResultadosESAR(os.path.dirname(r)))

print('-'*50)
print(f"ESAR 300: {SAR_300}")
print(f" tau 300: {tau_300}") 
print(f"  Hc 300: {Hc_300}")
print('-'*50)


SAR_268, tau_268, Hc_268 = extraer_SAR_tau(resultados_FR_268)
res_268=[]
print('Resultados 300 kHz', '='*80,'\n')
for r in resultados_FR_268:
    res_268.append(ResultadosESAR(os.path.dirname(r)))

print('-'*50)
print(f"ESAR 268: {SAR_268}")
print(f" tau 268: {tau_268}") 
print(f"  Hc 268: {Hc_268}")
print('-'*50)

SAR_213, tau_213, Hc_213 = extraer_SAR_tau(resultados_FR_213)
res_213=[]
print('Resultados 213 kHz', '='*80,'\n')
for r in resultados_FR_213:
    res_213.append(ResultadosESAR(os.path.dirname(r)))

print('-'*50)
print(f"ESAR 213: {SAR_213}")
print(f" tau 213: {tau_213}") 
print(f"  Hc 213: {Hc_213}")
print('-'*50)

SAR_135, tau_135, Hc_135 = extraer_SAR_tau(resultados_FR_135)
res_135=[]
print('Resultados 135 kHz', '='*80,'\n')
for r in resultados_FR_135:
    res_135.append(ResultadosESAR(os.path.dirname(r)))

print('-'*50)
print(f"ESAR 135: {SAR_135}")
print(f" tau 135: {tau_135}") 
print(f"  Hc 135: {Hc_135}")
print('-'*50)

SAR_081, tau_081, Hc_081 = extraer_SAR_tau(resultados_FR_081)
res_081=[]
print('Resultados 081 kHz', '='*80,'\n')
for r in resultados_FR_081:
    res_081.append(ResultadosESAR(os.path.dirname(r)))

print('-'*50)
print(f"ESAR 081: {SAR_081}")
print(f" tau 081: {tau_081}") 
print(f"  Hc 081: {Hc_081}")
print('-'*50)
#%% tau ESAR Hc vs H0
idc=[15,10,8,6,4,1.5]
H0 = [(h*pendiente_HvsI+ordenada_HvsI)/1000 for h in idc]

# idc_300=[15,10,8,6,4,3.5,3,2.5,2,1.5]
idc_300= [15,14.5,14,13.5,13,12.5,12,11.5,11,10.5,10,9.5,9,8.5,8,7.5,7,6.5,6,5.5,5,4.5,4,3.5,3,2.5,2,1.5,1.4]
H0_300 = [(h*pendiente_HvsI+ordenada_HvsI)/1000 for h in idc_300]


fig12,axs=plt.subplots(3,1,figsize=(10,8),constrained_layout=True)

axs[0].set_ylabel('tau (ns)')
axs[0].errorbar(H0_300,unumpy.nominal_values(tau_300),unumpy.std_devs(tau_300),fmt='.-',capsize=4,label='300',zorder=3)
axs[0].errorbar(H0,unumpy.nominal_values(tau_268),unumpy.std_devs(tau_268),fmt='.-',capsize=4,label='268')
axs[0].errorbar(H0,unumpy.nominal_values(tau_213),unumpy.std_devs(tau_213),fmt='.-',capsize=4,label='213')
axs[0].errorbar(H0,unumpy.nominal_values(tau_135),unumpy.std_devs(tau_135),fmt='.-',capsize=4,label='135')
axs[0].errorbar(H0,unumpy.nominal_values(tau_081),unumpy.std_devs(tau_081),fmt='.-',capsize=4,label='081')


axs[1].set_ylabel('ESAR (W/g)')
axs[1].errorbar(H0_300,unumpy.nominal_values(SAR_300),unumpy.std_devs(SAR_300),fmt='.-',capsize=4,label='300',zorder=3)
axs[1].errorbar(H0,unumpy.nominal_values(SAR_268),unumpy.std_devs(SAR_268),fmt='.-',capsize=4,label='268')
axs[1].errorbar(H0,unumpy.nominal_values(SAR_213),unumpy.std_devs(SAR_213),fmt='.-',capsize=4,label='213')
axs[1].errorbar(H0,unumpy.nominal_values(SAR_135),unumpy.std_devs(SAR_135),fmt='.-',capsize=4,label='135')
axs[1].errorbar(H0,unumpy.nominal_values(SAR_081),unumpy.std_devs(SAR_081),fmt='.-',capsize=4,label='081')

axs[2].set_ylabel('Hc (kA/m)')
axs[2].errorbar(H0_300,unumpy.nominal_values(Hc_300),unumpy.std_devs(Hc_300),fmt='.-',capsize=4,label='300',zorder=3)
axs[2].errorbar(H0,unumpy.nominal_values(Hc_268),unumpy.std_devs(Hc_268),fmt='.-',capsize=4,label='268')
axs[2].errorbar(H0,unumpy.nominal_values(Hc_213),unumpy.std_devs(Hc_213),fmt='.-',capsize=4,label='213')
axs[2].errorbar(H0,unumpy.nominal_values(Hc_135),unumpy.std_devs(Hc_135),fmt='.-',capsize=4,label='135')
axs[2].errorbar(H0,unumpy.nominal_values(Hc_081),unumpy.std_devs(Hc_081),fmt='.-',capsize=4,label='081')

for a in axs:
    a.grid()
    a.legend(title='$f$ (kHz)',loc='upper left',frameon=True,shadow=True,ncol=1)
    a.set_xticks([round(h2,1) for h2 in H0[::-1]])
    a.tick_params(axis='x')
axs[2].set_xlabel('H$_0$ (kA/m)')
plt.suptitle('Ferroresina\ntau / ESAR / H$_c$  vs H$_0$',fontsize=15)

# %% tau ESAR Hc vs f

f=[300,268,213,135,81]

tau_57=[tau_300[0],tau_268[0],tau_213[0],tau_135[0],tau_081[0]]
ESAR_57=[SAR_300[0],SAR_268[0],SAR_213[0],SAR_135[0],SAR_081[0]]
Hc_57=[Hc_300[0],Hc_268[0],Hc_213[0],Hc_135[0],Hc_081[0]] 
 
tau_38=[tau_300[10],tau_268[1],tau_213[1],tau_135[1],tau_081[1]]
ESAR_38=[SAR_300[10],SAR_268[1],SAR_213[1],SAR_135[1],SAR_081[1]]
Hc_38=[Hc_300[10],Hc_268[1],Hc_213[1],Hc_135[1],Hc_081[1]]

tau_31=[tau_300[14],tau_268[2],tau_213[2],tau_135[2],tau_081[2]]
ESAR_31=[SAR_300[14],SAR_268[2],SAR_213[2],SAR_135[2],SAR_081[2]]
Hc_31=[Hc_300[14],Hc_268[2],Hc_213[2],Hc_135[2],Hc_081[2]]

tau_23=[tau_300[18],tau_268[3],tau_213[3],tau_135[3],tau_081[3]]
ESAR_23=[SAR_300[18],SAR_268[3],SAR_213[3],SAR_135[3],SAR_081[3]]
Hc_23=[Hc_300[18],Hc_268[3],Hc_213[3],Hc_135[3],Hc_081[3]]

tau_16=[tau_300[22],tau_268[4],tau_213[4],tau_135[4],tau_081[4]]
ESAR_16=[SAR_300[22],SAR_268[4],SAR_213[4],SAR_135[4],SAR_081[4]]
Hc_16=[Hc_300[22],Hc_268[4],Hc_213[4],Hc_135[4],Hc_081[4]]

tau_07=[tau_300[27],tau_268[5],tau_213[5],tau_135[5],tau_081[5]]
ESAR_07=[SAR_300[27],SAR_268[5],SAR_213[5],SAR_135[5],SAR_081[5]]
Hc_07=[Hc_300[27],Hc_268[5],Hc_213[5],Hc_135[5],Hc_081[5]]

 
 
fig13,axs=plt.subplots(3,1,figsize=(10,8),constrained_layout=True)

axs[0].set_ylabel('tau (ns)')

axs[0].errorbar(f,unumpy.nominal_values(tau_57),unumpy.std_devs(tau_57),fmt='.-',capsize=4,label='57')
axs[0].errorbar(f,unumpy.nominal_values(tau_38),unumpy.std_devs(tau_38),fmt='.-',capsize=4,label='38')
axs[0].errorbar(f,unumpy.nominal_values(tau_31),unumpy.std_devs(tau_31),fmt='.-',capsize=4,label='31')  
axs[0].errorbar(f,unumpy.nominal_values(tau_23),unumpy.std_devs(tau_23),fmt='.-',capsize=4,label='23')
axs[0].errorbar(f,unumpy.nominal_values(tau_16),unumpy.std_devs(tau_16),fmt='.-',capsize=4,label='16')
axs[0].errorbar(f,unumpy.nominal_values(tau_07),unumpy.std_devs(tau_07),fmt='.-',capsize=4,label='07')

axs[1].set_ylabel('ESAR (W/g)')
axs[1].errorbar(f,unumpy.nominal_values(ESAR_57),unumpy.std_devs(ESAR_57),fmt='.-',capsize=4,label='57',zorder=3)
axs[1].errorbar(f,unumpy.nominal_values(ESAR_38),unumpy.std_devs(ESAR_38),fmt='.-',capsize=4,label='38')
axs[1].errorbar(f,unumpy.nominal_values(ESAR_31),unumpy.std_devs(ESAR_31),fmt='.-',capsize=4,label='31')
axs[1].errorbar(f,unumpy.nominal_values(ESAR_23),unumpy.std_devs(ESAR_23),fmt='.-',capsize=4,label='23')
axs[1].errorbar(f,unumpy.nominal_values(ESAR_16),unumpy.std_devs(ESAR_16),fmt='.-',capsize=4,label='16')
axs[1].errorbar(f,unumpy.nominal_values(ESAR_07),unumpy.std_devs(ESAR_07),fmt='.-',capsize=4,label='07')

axs[2].set_ylabel('Hc (kA/m)')
axs[2].errorbar(f,unumpy.nominal_values(Hc_57),unumpy.std_devs(Hc_57),fmt='.-',capsize=4,label='57',zorder=3)
axs[2].errorbar(f,unumpy.nominal_values(Hc_38),unumpy.std_devs(Hc_38),fmt='.-',capsize=4,label='38')
axs[2].errorbar(f,unumpy.nominal_values(Hc_31),unumpy.std_devs(Hc_31),fmt='.-',capsize=4,label='31')    
axs[2].errorbar(f,unumpy.nominal_values(Hc_23),unumpy.std_devs(Hc_23),fmt='.-',capsize=4,label='23')
axs[2].errorbar(f,unumpy.nominal_values(Hc_16),unumpy.std_devs(Hc_16),fmt='.-',capsize=4,label='16')
axs[2].errorbar(f,unumpy.nominal_values(Hc_07),unumpy.std_devs(Hc_07),fmt='.-',capsize=4,label='07')

for a in axs:
    a.grid()
    a.legend(title='$H_0$ (kA/m)',frameon=True,shadow=True,ncol=3)
    a.set_xticks(f)
    a.tick_params(axis='x')
axs[2].set_xlabel('Frecuencia (kHz)')

plt.suptitle('Ferroresina\ntau / ESAR / H$_c$  vs Frecuencia',fontsize=15)
# %% salvo figuras de los resultados
fig12.savefig('04_tau_ESAR_Hc_vs_H0.png')
fig13.savefig('05_tau_ESAR_Hc_vs_f.png')
# %%

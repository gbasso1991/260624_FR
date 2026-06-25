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
            "pendiente_HvsI ": float(lines[3].strip().split('_=_')[1].split(' ')[0]),
            "ordenada_HvsI ": float(lines[4].strip().split('_=_')[1].split(' ')[0]),
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
    return SAR, tau, Hc
#%% Ciclos y resultados para 268 , 213 135 y 081 
conc_FR =  20 #g/L

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
#plt.savefig('0_ciclos_promedio_FR_300kHz.png',dpi=300)

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
#%%
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

#%% comparo las 4 medidas

fig04, axs =plt.subplots(2,2,figsize=(10,7.5),constrained_layout=True,sharey=True,sharex=True)

axs[0,0].set_title('268 kHz',loc='left')
axs[0,1].set_title('213 kHz',loc='left')
axs[1,0].set_title('135 kHz',loc='left')
axs[1,1].set_title('081 kHz',loc='left')
for i,e in enumerate(ciclos_FR_268):
    _,_,_, H_FR,M_FR,_ = lector_ciclos(ciclos_FR_268[i])
    axs[0,0].plot(H_FR/1000,M_FR,label=f'{H0[i]:.1f}')
for i,e in enumerate(ciclos_FR_213):
    _,_,_, H_FR,M_FR,_ = lector_ciclos(ciclos_FR_213[i])
    axs[0,1].plot(H_FR/1000,M_FR,label=f'{H0[i]:.1f}')
for i,e in enumerate(ciclos_FR_135):
    _,_,_, H_FR,M_FR,_ = lector_ciclos(ciclos_FR_135[i])
    axs[1,0].plot(H_FR/1000,M_FR,label=f'{H0[i]:.1f}')
for i,e in enumerate(ciclos_FR_081):
    _,_,_, H_FR,M_FR,_ = lector_ciclos(ciclos_FR_081[i])
    axs[1,1].plot(H_FR/1000,M_FR,label=f'{H0[i]:.1f}')
    
    
for a in axs.flatten():
    a.grid()
    a.legend(loc='upper left',frameon=True,shadow=True,title='H$_0$ (kA/m)',ncol=2)
plt.suptitle(f'Ciclos promedio FR  [57; 6.9] kA/m')
# %%

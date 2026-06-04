import matplotlib.pyplot as plt
import numpy as np

tamanhos = ['256x256', '512x512', '1024x1024', '2048x2048']

naive      = [0.82, 0.72, 0.46, 0.30]  
cap3_simd  = [9.78, 8.91, 7.96, 3.93] 
cap4_unrol = [11.62, 11.62, 10.35, 4.25] 
cap5_block = [6.69, 6.53, 6.42, 8.47] 
cap6_omp   = [4.03, 21.44, 28.44, 20.19]

x = np.arange(len(tamanhos))
width = 0.15

fig, ax = plt.subplots(figsize=(12, 7))

rects1 = ax.bar(x - 2*width, naive, width, label='1. Baseline', color='#e63946')
rects2 = ax.bar(x - width, cap3_simd, width, label='2. Cap 3 (SIMD)', color='#f4a261')
rects3 = ax.bar(x, cap4_unrol, width, label='3. Cap 4 (SIMD+Unroll)', color='#2a9d8f')
rects4 = ax.bar(x + width, cap5_block, width, label='4. Cap 5 (Block)', color='#e9c46a')
rects5 = ax.bar(x + 2*width, cap6_omp, width, label='5. Cap 6 (OMP)', color='#264653')

ax.set_ylabel('Desempenho (GFLOPS)', fontsize=12, fontweight='bold')
ax.set_xlabel('Tamanho da Matriz', fontsize=12, fontweight='bold')
ax.set_title('Evolução DGEMM - Metodologia "Going Faster" (Patterson & Hennessy)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(tamanhos)
ax.legend(loc='upper left')
ax.grid(axis='y', linestyle='--', alpha=0.7)

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        if height > 0:
            ax.annotate(f'{height:.1f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)
autolabel(rects5)

fig.tight_layout()
plt.savefig('grafico_gflops.png', dpi=300)
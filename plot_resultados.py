import matplotlib.pyplot as plt
import numpy as np

# ==============================================================================
# SEUS RESULTADOS REAIS (GFLOPS)
# ==============================================================================
# Ajustado para 3 tamanhos para bater exatamente com as suas listas de dados
tamanhos = ['256x256', '512x512', '1024x1024']

# Seus dados:
naive     = [0.80, 0.66, 0.41]  
reordered = [14.79, 15.33, 11.11]  
simd      = [15.67, 13.87, 11.03] 
unroll    = [14.88, 13.07, 10.13] 
final_omp = [4.10, 22.55, 28.05] 

# ==============================================================================
# CONFIGURAÇÃO DO GRÁFICO 
# ==============================================================================
x = np.arange(len(tamanhos))  # Localização das labels
width = 0.15  # Largura das barras

fig, ax = plt.subplots(figsize=(12, 7))

rects1 = ax.bar(x - 2*width, naive, width, label='1. Naive (Cap. Baseline)', color='#e63946')
rects2 = ax.bar(x - width, reordered, width, label='2. Reordered (Localidade)', color='#f4a261')
rects3 = ax.bar(x, simd, width, label='3. SIMD AVX (Cap. 3)', color='#2a9d8f')
rects4 = ax.bar(x + width, unroll, width, label='4. SIMD + Unroll (Cap. 4)', color='#e9c46a')
rects5 = ax.bar(x + 2*width, final_omp, width, label='5. SIMD+Block+OMP (Cap. 5 e 6)', color='#264653')

# Adicionando textos e labels
ax.set_ylabel('Desempenho (GFLOPS)', fontsize=12, fontweight='bold')
ax.set_xlabel('Tamanho da Matriz', fontsize=12, fontweight='bold')
ax.set_title('Evolução do Desempenho DGEMM - Metodologia "Going Faster"', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(tamanhos)
ax.legend(loc='upper left')
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Função para colocar o valor exato em cima de cada barra
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        if height > 0:
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)
autolabel(rects5)

fig.tight_layout()

# Salva a imagem na mesma pasta
plt.savefig('grafico_gflops.png', dpi=300)
print("Gráfico gerado com sucesso: grafico_gflops.png")
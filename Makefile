# ==========================================
# Configurações do Compilador
# ==========================================
CXX = g++

# Flags de Otimização:
# -O3          : Nível máximo de otimização segura do compilador.
# -march=native: Permite usar instruções vetoriais (AVX/SSE) do SEU processador.
# -Wall        : Mostra todos os avisos do compilador.
# -std=c++11   : Garante compatibilidade com a biblioteca <chrono>.
CXXFLAGS = -O3 -march=native -Wall -std=c++11

# Nome do arquivo executável que será gerado
TARGET = benchmark

# ==========================================
# Regras de Compilação
# ==========================================

# Regra padrão executada quando você digita apenas 'make'
all: $(TARGET)

# Como construir o executável
$(TARGET): DgemmCompleto.cpp
	$(CXX) $(CXXFLAGS) DgemmCompleto.cpp -o $(TARGET)

# Regra para limpar os arquivos compilados (útil antes de um novo commit)
clean:
	rm -f $(TARGET)

# Regra para compilar e já executar o código automaticamente
run: $(TARGET)
	./$(TARGET)
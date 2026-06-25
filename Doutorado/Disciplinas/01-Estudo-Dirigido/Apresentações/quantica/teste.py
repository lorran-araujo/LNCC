from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram

# 1. Criar um Circuito Quântico com 1 qubit e 1 bit clássico
# (O bit clássico serve para guardar o resultado da medição)
qc = QuantumCircuit(1, 1)

# 2. Aplicar a porta Hadamard (H) no qubit 0
# Isso tira o qubit do estado |0> e o coloca em sobreposição
qc.h(0)

# 3. Medir o qubit
# O resultado do qubit 0 será armazenado no bit clássico 0
qc.measure(0, 0)

# 4. Simular o circuito
# Usamos o 'qasm_simulator' que imita um computador quântico ideal
backend = Aer.get_backend('qasm_simulator')
job = backend.run(qc, shots=1000) # Rodamos o experimento 1000 vezes
result = job.result()

# 5. Mostrar os resultados
counts = result.get_counts(qc)
print(f"Resultados das 1000 execuções: {counts}")
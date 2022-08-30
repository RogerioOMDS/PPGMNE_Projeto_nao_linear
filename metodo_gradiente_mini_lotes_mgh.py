import numpy as np 
import pyopus.problems.mgh as problems
import random 
import time

# Não resolve GaoHanAlmostQuadratic, não tem Jacobiana, m = 1
# Não resolve McKinnon, não tem Jacobiana, m = 1

def metodo_gradiente_mini_lote(funcao, x, lote = 1, max_iter = 500000, epsilon = 1e-3, t = 1e-3, tempo_limite = 108000):

    f, g = funcao.fg(x)
    norma = np.linalg.norm(g)
    m = funcao.m 


    if (lote > m): 
        print("Atençã, este tamanho de lote não é aceitável.")
        return "Atente para lote < m."

    if (m == 0): 
        return ("Atenção, esta função tem m = 0, portanto não aceita lote.")

    lotes = np.ceil(m / lote)
    indices = np.arange(m)
    random.shuffle(indices)
    intervalos = np.array_split(indices, lotes)

    iter = 0
    start_time = time.time()
    delta_tempo = time.time() - start_time
    while ((iter < max_iter) and (norma > epsilon) and (delta_tempo < tempo_limite)):


        J = funcao.J
        fi = funcao.fi
        
        for lista_indices in intervalos:
            Ji = J[lista_indices, :]
            fii = fi[lista_indices].reshape(-1,1)
            vetor = 2 * np.multiply(Ji, fii).sum(axis = 0)  / len(lista_indices)
            x = x - t * vetor

        f, g = funcao.fg(x)

        norma = np.linalg.norm(g)
        iter += 1
        delta_tempo = time.time() - start_time

        # print("norma = ", norma) 

    end_time = time.time()
    delta_tempo = (end_time - start_time)

    f, g = funcao.fg(x)

    if (norma < epsilon):
        status = "sucesso"
    elif (iter >= max_iter):
        status = "iter_max"
    elif (delta_tempo >= tempo_limite):
        status = "time_max"
    else: 
        status = "erro"    

    resultado = {
        "status": status,
        "x" : x, 
        "iter": iter, 
        "objetivo": f, 
        "gradiente": g, 
        "norma": norma,
        "tempo (s)": delta_tempo, 
        "lote": lote, 
        "t": t
    }

    return resultado

# funcao = problems.Rosenbrock()
# x = funcao.initial


# aplicacao = metodo_gradiente_mini_lote(funcao, x)

# print(aplicacao)
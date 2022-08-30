import pyopus.problems.mgh as problems
import numpy as np 
import random 
import time


def metodo_gradiente_estocastico(funcao, x, max_iter = 1000000, epsilon = 1e-3, t = 1e-3, tempo_limite = 108000, lote = None):

    f, g = funcao.fg(x)
    norma = np.linalg.norm(g)

    indices = np.arange(funcao.m)
    random.shuffle(indices)

    iter = 0
    start_time = time.time()
    delta_tempo = time.time() - start_time
    while ((iter < max_iter) and (norma > epsilon) and (delta_tempo < tempo_limite)):

        J = funcao.J
        fi = funcao.fi
        
        for indice in indices:
            Ji = J[indice, :]
            fii = fi[indice]
            x = x - t * 2 * np.multiply(Ji, fii)  

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
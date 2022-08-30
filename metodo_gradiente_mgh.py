import numpy as np
import pyopus.problems.mgh as problems
import time

def metodo_gradiente(funcao, x, max_iter = 1000000, epsilon = 1e-3, t = 1e-3, tempo_limite = 108000, lote = None):
  
    g = funcao.g(x)
    norma = np.linalg.norm(g)

    iter = 0
    start_time = time.time()
    delta_tempo = time.time() - start_time
    while ((iter < max_iter) and (norma > epsilon) and (delta_tempo < tempo_limite)):

        x = x - t * g 

        g = funcao.g(x)
        norma = np.linalg.norm(g)
        iter += 1
        delta_tempo = time.time() - start_time
        
        # print("norma = ", norma)

    end_time = time.time()

    delta_tempo = (end_time - start_time)

    f = funcao.f(x)

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

# funcao = problems.BrownBadlyScaled()
# x = funcao.initial


# aplicacao = metodo_gradiente(funcao, x, t= 1e-4)

# print(aplicacao)
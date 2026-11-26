# Block-Diagonal LoRA for Eliminating Communication Overhead in Tensor Parallel LoRA Serving — BD-LoRA (NeurIPS 2025, University of Warwick / AWS — Wang et al.)

**Pregunta con la que lo leí:** ¿Cómo elimina BD-LoRA la comunicación extra entre GPUs al servir muchos adaptadores LoRA con *tensor parallelism*, y qué queda de él al entrenarlo en una sola GPU en `FLUX.2-klein`?

**Problema:** Al servir múltiples adaptadores LoRA distintos sobre un mismo modelo base no se pueden fusionar con los pesos (cambiar de adaptador obligaría a restar y sumar pesos, y no se podrían agrupar en lote peticiones con adaptadores distintos), así que los cálculos de LoRA van aparte ($XW + XAB$). Cuando el modelo se reparte en varias GPUs (*tensor parallelism*, como hace S-LoRA), las tarjetas tienen que pasarse datos extra (*all-gather* / *all-reduce*) para completar las multiplicaciones de LoRA, además de la comunicación que ya hace el modelo base, y ese tráfico penaliza mucho la velocidad de inferencia aunque en teoría sea pequeño.

**Idea clave:** Forzar a que ciertas matrices del adaptador ($B_1$ y $A_2$) sean **bloque-diagonales**: esos ceros actúan como muros aislantes que permiten que cada GPU ejecute su parte del producto matricial (GEMM) solo con su memoria local, sin ninguna comunicación adicional; la pérdida de expresividad se compensa subiendo $r$.

**Método:**
Una capa típica de Transformer tiene dos transformaciones, $W_1$ y $W_2$, que el modelo base reparte entre GPUs siguiendo Megatron-LM: $W_1$ cortada por columnas y $W_2$ cortada por filas. BD-LoRA alinea el adaptador con ese reparto:

```math
XW_1 + XA_1B_1, \quad B_1 = \text{diag}\big(B_1^{(1)}, \dots, B_1^{(N)}\big); \qquad YW_2 + YA_2B_2, \quad A_2 = \text{diag}\big(A_2^{(1)}, \dots, A_2^{(N)}\big)
```

* **Primera transformación ($W_1$, por columnas):** se corta $A_1$ por columnas y se fuerza $B_1$ a ser bloque-diagonal. Así el resultado intermedio de LoRA ($XA_1B_1$) queda cortado por columnas exactamente igual que el del modelo base ($XW_1$), y como ambos trozos están en la misma GPU se suman sin enviar ni recibir datos de otras tarjetas.
* **Segunda transformación ($W_2$, por filas):** se fuerza $A_2$ a ser bloque-diagonal y se corta $B_2$ por filas. Cada GPU calcula su parte con la entrada que ya tiene y el *all-reduce* que el modelo base hace de todos modos suma también la contribución de LoRA. Se elimina por completo la sobrecarga de comunicación de LoRA.
* **N mini-LoRAs y compensación:** una matriz bloque-diagonal repartida en $N$ GPUs es matemáticamente idéntica a tener $N$ mini-adaptadores LoRA independientes de rango $r/N$, uno por trozo del modelo base (así se implementa en `peft`). Llenar la matriz de ceros le quita expresividad (a igual $r$ tiene más o menos la mitad de parámetros), y se soluciona aumentando $r$ para que los parámetros efectivos (no nulos) igualen a los de un LoRA estándar. El paper usa la escala de rsLoRA, porque con ella el rendimiento sube con $r$.

**Resultado clave:**
* **Calidad (Llama-3.2-1B y Llama-3.1-8B, GLUE y OpenOrca):** con un número parecido de parámetros, BD-LoRA iguala a LoRA y a menudo lo supera, aunque los intervalos de confianza se solapan mucho.
* **Velocidad (vLLM, 8 A100):** hasta **1.79x** más rápido que S-LoRA con 0.87x los parámetros del adaptador en Llama-3.1-70B, y hasta 1.63x en Llama-3.1-8B. A igual calidad, 1.27x–1.51x en Llama-3.1-8B: domina en Pareto a S-LoRA.
* **Mis experimentos (`FLUX.2-klein`, `nblocks=4`, una sola GPU):** con $r=32$ y la escala estándar da **DINOv2 = 0.7102** / drift 0.2626 frente a LoRA $r=32$ (0.6815 / 0.2721), con **15,482,880** parámetros (0.40%, un 60% menos), un checkpoint de 59.1 MB frente a 146.3 MB, 10,160 MB de memoria y 1,140 s. La config más cercana en parámetros a LoRA ($r=64$, 30.9M) da 0.6820 frente a 0.6815: coincide con lo que dice el paper.

**Limitaciones:**
* **Qué no prueban (y qué no mide mi experimento):** el paper solo evalúa LLMs (Llama) en GLUE y OpenOrca. En `run.py` se entrena un único adaptador en una sola GPU, así que la ventaja de velocidad de BD-LoRA, que es su razón de ser, no se mide: solo la calidad de un LoRA con la restricción bloque-diagonal. Con una sola semilla y 4 imágenes de test, parte de la diferencia en DINOv2 (+0.029) puede ser ruido.
* **Supuestos:** hay que saber en cuántas GPUs ($N$) se va a servir el modelo desde el día en que se empieza a entrenar, porque se entrenan literalmente $N$ mini-adaptadores aislados (los autores argumentan que suele haber un grado de *tensor parallelism* típico por modelo y hardware). Y la escala importa: con la estándar, duplicar el rango no mejora el DINOv2 ($r=64$: 0.6820 frente a 0.7102 con $r=32$), como también ve el paper.
* **Coste (escala rsLoRA):** en mis runs, la escala del paper empeoró ($r=64$: 0.6964 / drift 0.2872; $r=32$: 0.6092 / drift 0.3104, el peor de los cuatro): con `lora_alpha=64`, rsLoRA multiplica la escala por 8 ($r=64$) y por 11.3 ($r=32$) respecto a la estándar, con el lr en `1e-4`, lo que probablemente fue demasiado agresivo. Para servir de verdad hace falta soporte en el servidor (el de vLLM está como *draft PR*).

**Conexión:** `#peft` `#lora` `#serving` `#multi-lora` `#tensor-parallelism` `#vllm` `#rslora` `#flux2-klein` (método pensado para inferencia multi-adaptador, no para entrenar mejor: el código de entrenamiento está en `peft` y el de *serving* como *draft PR* en vLLM; en el benchmark de imagen solo se ve su efecto como regularizador estructural de LoRA).

**Preguntas**
1. **Conceptual:** *¿Por qué no se fusionan los adaptadores al servir muchos a la vez, y de dónde sale la comunicación extra de S-LoRA?*
   * **Respuesta:** Porque cambiar de adaptador obligaría a restar los pesos de LoRA anteriores y sumar los nuevos, y no se podrían batchear peticiones que usan adaptadores distintos; por eso se calcula $XW + XAB$ por separado. Con *tensor parallelism*, S-LoRA trocea $A$ y $B$ entre GPUs alineado con Megatron-LM, pero para multiplicar por la segunda matriz densa cada GPU necesita los trozos de las demás: hace falta un *all-gather* / *all-reduce* extra además del del modelo base.
2. **Matemática / Diseño:** *¿Qué matrices hace bloque-diagonales BD-LoRA y por qué precisamente esas?*
   * **Respuesta:** $B_1$ (con $A_1$ cortada por columnas) y $A_2$ (con $B_2$ cortada por filas). Con $B_1$ bloque-diagonal, cada GPU obtiene su trozo de columnas de $XA_1B_1$, el mismo que tiene de $XW_1$; con $A_2$ bloque-diagonal, cada GPU procesa solo su trozo de la entrada y su resultado parcial entra en el *all-reduce* que ya hace el modelo base. Es lo mismo que poner $N$ LoRAs independientes de rango $r/N$, uno por trozo de pesos.
3. **Crítica / Trade-offs:** *BD-LoRA superó a LoRA en `FLUX.2-klein` con un 60% menos de parámetros. ¿Prueba eso la tesis del paper?*
   * **Respuesta:** No del todo: la tesis es que, con parámetros efectivos parecidos, iguala la calidad de LoRA y es mucho más rápido al servir en varias GPUs. Mi experimento entrena un solo adaptador en una GPU, así que la velocidad no se mide. En calidad, la comparación justa ($r=64$, 30.9M) da 0.6820 frente a 0.6815, que sí coincide con el paper; el +0.029 de $r=32$ es una sola semilla con 4 imágenes de test.

**Flashcards**
1. **P:** ¿Qué problema resuelve BD-LoRA?
   **R:** La comunicación extra entre GPUs (*all-gather* / *all-reduce*) que necesita S-LoRA para calcular los adaptadores LoRA cuando se sirven muchos a la vez con *tensor parallelism*.
2. **P:** ¿Qué matrices del adaptador son bloque-diagonales en BD-LoRA?
   **R:** $B_1$ (en la transformación cortada por columnas) y $A_2$ (en la cortada por filas), alineadas con el reparto Megatron-LM del modelo base.
3. **P:** ¿Cuál es la principal limitación práctica de BD-LoRA?
   **R:** Equivale a $N$ mini-adaptadores (uno por GPU), así que hay que conocer el número de GPUs de *serving* ($N$) desde que se empieza a entrenar.
4. **P:** ¿Por qué en mis runs subir $r$ con la escala estándar no ayudó y rsLoRA empeoró?
   **R:** Con la escala estándar, LoRA y BD-LoRA apenas mejoran con $r$. Con `lora_alpha=64`, rsLoRA multiplicó la escala por 8–11.3 manteniendo el lr en `1e-4`, demasiado agresivo (sobre todo con $r=32$).

# Parameter-Efficient Fine-Tuning with Discrete Fourier Transform — FourierFT (ICML 2024, HKUST (Guangzhou) / HKUST / Sun Yat-sen University — Gao et al.)

**Pregunta con la que lo leí:** ¿Cómo aprende FourierFT el cambio de pesos con unos pocos coeficientes de frecuencia, y por qué en `FLUX.2-klein` necesitó un *learning rate* 3,000 veces mayor que el por defecto?

**Problema:** Aunque LoRA reduce mucho los parámetros, sigue siendo caro de almacenar y de cargar cuando hay muchas adaptaciones personalizadas o modelos base grandes, y su número de parámetros crece con el tamaño de cada capa ($r \cdot (\text{in} + \text{out})$).

**Idea clave:** Entrenar el cambio de pesos en el dominio de la frecuencia en lugar de directamente sobre la matriz: $\Delta W$ se ve como una señal 2D (como una imagen) y, igual que en la compresión de imágenes, basta con unos pocos coeficientes de frecuencia para reconstruirla.

**Método:**
FourierFT trata $\Delta W$ como una matriz en el dominio espacial y solo aprende una pequeña fracción de sus coeficientes espectrales; después aplica la transformada inversa de Fourier 2D para recuperarla:

$$
F_{ij} = \begin{cases} c_k & \text{si } (i,j) = E_k \\ 0 & \text{en el resto} \end{cases}, \qquad \Delta W = \alpha \cdot \text{Re}\big(\text{IDFT}_{2}(F)\big), \qquad h = W_0 x + \Delta W x
$$

* **Posiciones fijas:** para cada capa se eligen al azar $n$ posiciones de una matriz del mismo tamaño que $W$ (`n_frequency`, 2000 en mi experimento). Las posiciones se fijan con una semilla (`random_loc_seed`) y no se entrenan: en el paper se comparten entre todas las capas, y en `peft` las capas con la misma forma acaban con las mismas posiciones.
* **Coeficientes entrenables:** lo único que se entrena son $n$ coeficientes por capa, uno por posición; el resto de la matriz de frecuencias $F$ vale 0. En cada paso se construye esa matriz dispersa, se le aplica la transformada inversa de Fourier 2D (`ifft2`) y la parte real, multiplicada por `scaling` ($\alpha$, 150 por defecto en `peft`; el paper usa 300 en casi todo), se suma a los pesos. En el paper los coeficientes se inicializan con una gaussiana estándar, no a 0, así que $\Delta W$ no es exactamente cero al empezar; en `peft` eso es `init_weights=False`, el valor de mi config (con `True` empezaría a 0).
* **Por qué bastan tan pocos:** cada coeficiente no toca un solo peso: es una onda que se extiende por toda la matriz, así que con muy pocos parámetros se modifica la matriz entera. Y la base importa: sustituir la de Fourier por una base aleatoria u ortogonal empeora (RTE con RoBERTa-base: 79.1 frente a 72.7 y 75.6). El número de parámetros no depende del tamaño de la capa, solo de $n$: en mi experimento son 2000 × 80 capas = **160,000**, unas 240 veces menos que LoRA $r=32$ (38.3M). Al terminar se puede fusionar con los pesos (*merge*), así que en inferencia no añade coste.

**Resultado clave:**
* **GLUE (RoBERTa):** en base, 85.0 de media con **0.024M** parámetros frente a 85.2 de LoRA con 0.3M; en large, **88.0** con 0.048M frente a 87.8 con 0.8M. Subir $n$ mejora de forma consistente, mientras que subir $r$ en LoRA no siempre ayuda.
* **Instruction tuning (LLaMA2, MT-Bench / Vicuna):** en 7B, 5.18 / 7.49 con **0.064M** frente a 5.20 / 7.35 de LoRA con **33.5M**; en 13B, 5.82 / 7.92 con 0.08M frente a 5.80 / 7.89 con 52.4M. Las diferencias quedan dentro de la desviación (±0.3), así que es un empate con 500 veces menos parámetros.
* **Visión (ViT-Base, 8 datasets):** 80.29% con 239K parámetros frente a 77.58% de LoRA con 581K (*full fine-tuning*: 86.49%).
* **Generación por sujeto (DreamBooth en SD1.5, apéndice):** FID de 244.9 con **0.19M** parámetros frente a 245.2 de LoRA con 12.4M (más bajo es mejor; *full fine-tuning*: 221.6).
* **Mis experimentos (`FLUX.2-klein`, $n=2000$, `scaling=150`):** con el lr actual (`1e-4`) solo llegaba a **0.347** de DINOv2 (drift 0.021) y la similitud de validación se quedaba plana en ~0.236 todo el entrenamiento: el modelo no estaba aprendiendo. Barrido de lr con el mismo `adapter_config.json`: `1e-2` → 0.401 / 0.047; `1e-1` → 0.570 / 0.144; **`3e-1` → 0.625** / 0.249, con el drift ya al nivel de LoRA (0.262 en la L40S). Propuse actualizar el lr a `3e-1` con un `training_params.json` (PR #3933, ya mergeado).

**Limitaciones:**
* **Qué no prueban:** en todos los experimentos solo adapta las matrices *query* y *value* (en FLUX adapté los 11 tipos de módulo). En difusión solo hay un experimento pequeño en el apéndice (DreamBooth en SD1.5, medido con FID). No hay sección de limitaciones ni comparación de tiempo de entrenamiento: sus ventajas se miden en parámetros y almacenamiento, no en cómputo. Hacer entrenable la frecuencia central (que ahora habría que buscar a mano) queda para el futuro.
* **Supuestos (*learning rate*):** en GLUE, E2E y ViT (capas de 768 a 1024) el paper usa learning rates entre `1e-2` y `3e-1`. El motivo: `torch.fft.ifft2` divide por el número de elementos de la matriz (out × in). En `FLUX.2-klein` las capas son de 3072 × 3072 o más (9.4M de elementos o más), frente a los 768 × 768 (0.6M) de RoBERTa-base, así que cada coeficiente pesa unas 16 veces menos en cada peso; con Adam, cada paso mueve los coeficientes más o menos lo mismo (del orden del lr), y $\Delta W$ crece muy despacio si el lr no es muy alto. Matiz: en LLaMA2 (capas de 4096 × 4096, más grandes que las de FLUX) el paper usa `3e-2`, pero con $\alpha = 300$ (el doble del default de `peft`) y una época completa de Alpaca, muchos más pasos que los 750 del benchmark: el lr necesario depende también de `scaling` y del número de pasos. Subir `scaling` debería tener un efecto parecido al lr (no lo he probado), y no probé valores por encima de `3e-1`, así que no sé si seguiría mejorando o rompería el modelo.
* **Coste:** es lento: en cada paso construye la matriz completa $\Delta W$ con la `ifft2`. En el benchmark oficial (L40S) entrena en 2,988 s frente a 759 s de LoRA, unas 4 veces más, y ese tiempo no depende del lr.

**Conexión:** `#peft` `#fourier` `#frequency-domain` `#extreme-peft` `#learning-rate` `#flux2-klein` (como HiRA y LoHa, pierde el atajo de LoRA y construye $\Delta W$ entera en cada paso. Con 160,000 parámetros queda cerca de métodos de pocos parámetros como Uni-LoRA (0.6573 con 500,000) o VeRA (0.6627 con 893,440), aunque esos dos son de la L40S. La run de `1e-2` no pude proponerla para el graveyard por un bug al escribir los resultados, que reporté en el issue #3921).

**Preguntas**
1. **Conceptual:** *¿Por qué FourierFT puede modificar toda la matriz de pesos entrenando unos pocos coeficientes, y por qué sus parámetros no dependen del tamaño de la capa?*
   * **Respuesta:** Porque cada coeficiente de frecuencia, al pasar por la transformada inversa, se convierte en una onda que se extiende por toda la matriz: con $n$ coeficientes se obtiene una $\Delta W$ densa del tamaño completo de $W$. Lo entrenable son solo esos $n$ coeficientes por capa (las posiciones son fijas y compartidas), así que el coste en parámetros es $n \times L$, sin importar las dimensiones de la capa.
2. **Matemática / Diseño:** *¿Por qué FourierFT necesitó un lr de `3e-1` en `FLUX.2-klein` y qué tiene que ver la `ifft2`?*
   * **Respuesta:** `ifft2` normaliza dividiendo por out × in. En las capas de FLUX (≥ 3072 × 3072) eso hace que cada coeficiente aporte unas 16 veces menos a cada peso que en RoBERTa-base. Como Adam mueve cada coeficiente del orden del lr por paso, con `1e-4` $\Delta W$ casi no cambia en 750 pasos (drift 0.021, igual que el modelo base). Se compensa subiendo el lr, `scaling` o el número de pasos (en LLaMA2 el paper usa `3e-2` con $\alpha = 300$ y una época de Alpaca).
3. **Crítica / Trade-offs:** *FourierFT entrena unas 240 veces menos parámetros que LoRA en `FLUX.2-klein`. ¿Por qué tarda unas 4 veces más?*
   * **Respuesta:** Porque pocos parámetros no implican poco cómputo: en cada paso tiene que construir la matriz de frecuencias completa, aplicar la `ifft2` sobre matrices de millones de elementos y hacer la multiplicación con $\Delta W$ densa, mientras que LoRA nunca construye la matriz completa. Su ventaja está en el almacenamiento y la memoria de los adaptadores, no en la velocidad de entrenamiento.

**Flashcards**
1. **P:** ¿Cómo calcula FourierFT el cambio de pesos?
   **R:** $\Delta W = \alpha \cdot \text{Re}(\text{IDFT}_2(F))$, donde $F$ es una matriz de frecuencias con $n$ coeficientes entrenables en posiciones aleatorias fijas y ceros en el resto.
2. **P:** ¿Qué se entrena y qué se fija en FourierFT?
   **R:** Se entrenan $n$ coeficientes por capa; las posiciones (semilla `random_loc_seed`) y `scaling` son fijas y compartidas.
3. **P:** ¿Qué pasó al subir el lr de FourierFT en `FLUX.2-klein`?
   **R:** De `1e-4` (0.347, no aprendía) a `3e-1` (0.625, drift al nivel de LoRA); la mejora fue monótona y propuse `3e-1` como nueva entrada (PR #3933).
4. **P:** ¿Por qué FourierFT es lento pese a tener pocos parámetros?
   **R:** Porque en cada paso construye la $\Delta W$ completa con la `ifft2` (2,988 s frente a 759 s de LoRA en la L40S).

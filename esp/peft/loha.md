# FedPara: Low-Rank Hadamard Product for Communication-Efficient Federated Learning — LoHa (ICLR 2022, POSTECH — Hyeon-Woo et al.)

**Pregunta con la que lo leí:** ¿Cómo consigue LoHa rango $r^2$ multiplicando dos productos de bajo rango, por qué necesitó un *learning rate* mayor que LoRA en `FLUX.2-klein`, y le compensa frente a un LoRA con los mismos parámetros?

**Problema:** En aprendizaje federado, subir y bajar el modelo en cada ronda es caro, y la parametrización de bajo rango clásica reduce los parámetros pero limita la expresividad (rango $\le r$), así que no llega al rendimiento del modelo original. En PEFT pasa lo mismo: LoRA con $r=32$ está limitado a rango 32.

**Idea clave:** En vez de un único par de matrices como LoRA, usar **dos pares** y combinar sus productos con el producto Hadamard (elemento por elemento): cada producto tiene rango $\le r$, pero su Hadamard puede llegar a $r^2$, así que con pocos parámetros se obtiene una matriz de rango mucho mayor.

**Método:**
LoHa (*low-rank Hadamard product*) viene de FedPara, un paper de aprendizaje federado, y se popularizó con LyCORIS para Stable Diffusion. En FedPara es la propia matriz de pesos la que se parametriza así, $W = (X_1Y_1^\top) \odot (X_2Y_2^\top)$, y se entrena desde cero; LoHa aplica la misma idea a la actualización $\Delta W$ sobre pesos congelados:

```math
\Delta W = (A_1 B_1) \odot (A_2 B_2) \cdot \frac{\alpha}{r}, \qquad \text{rango}(\Delta W) \le \text{rango}(A_1 B_1) \cdot \text{rango}(A_2 B_2) \le r^2
```

* **La clave es el rango:** cada producto $A_i B_i$ tiene como mucho rango $r$, pero su Hadamard puede llegar a rango $r^2$ (la misma desigualdad que usa HiRA). Con $r=32$, LoRA está limitado a rango 32 y LoHa puede llegar a **1024**, con el doble de parámetros ($2 \cdot r \cdot (\text{in} + \text{out})$ por capa frente a $r \cdot (\text{in} + \text{out})$). Para llegar a ese rango, LoRA necesitaría $r=1024$, con 16 veces más parámetros que LoHa. Si $r^2 \ge \min(m, n)$, la matriz puede llegar (con alta probabilidad) a rango completo, y para un rango objetivo $R^2$ usar el mismo $R$ en los dos pares es la elección con menos parámetros (Proposición 2), que es lo que hace `peft` con un único `r`.
* **Diferencia con HiRA:** usa el producto Hadamard igual que HiRA, pero en vez de multiplicar por los pesos originales $W_0$, multiplica entre sí dos actualizaciones de bajo rango: no se apoya en el conocimiento de $W_0$, pero tampoco depende de él.
* **Inicialización y coste:** igual que LoRA, empieza sin modificar el modelo: en `peft` tres de las cuatro matrices se inicializan aleatoriamente y la cuarta (`hada_w2_b`) a ceros, así que el segundo producto vale 0 y $\Delta W = 0$ al principio. Pierde el atajo de LoRA, que multiplica la entrada por $A$ y luego por $B$ sin construir nunca la matriz completa: el Hadamard obliga a construir $\Delta W$ entera en cada paso, igual que HiRA. Al terminar se puede fusionar con los pesos (*merge*), así que en inferencia no añade coste.

**Resultado clave:**
* **FedPara (aprendizaje federado):** VGG16 con FedPara alcanza la misma precisión objetivo que el VGG16 original con entre **2.8 y 10.1 veces** menos coste de comunicación y de energía. Con los mismos parámetros (~10% de VGG16), supera a la parametrización de bajo rango clásica (CIFAR-100 IID: **45.78%** frente a 34.16%), y lo mismo pasa con LSTM. Su versión personalizada (pFedPara) supera a otros métodos personalizados con más de 3 veces menos parámetros.
* **Mis experimentos (`FLUX.2-klein`, $r=32$, `alpha=32`, 76.7M parámetros):** con el lr por defecto (`1e-4`, L40S) daba **0.662** / drift 0.260. Probé tres learning rates con el mismo `adapter_config.json`: `3e-4` → 0.684 / 0.255; **`5e-4` → 0.707 / 0.256**; `1e-3` → 0.697 / 0.275 (ya se pasa: sube el drift y baja el DINOv2). Elegí `5e-4` y `1e-3` a partir del experimento ajustado de LoKr (lr `1e-3`) y añadí `3e-4` al ver que `5e-4` mejoraba a `1e-3`. Propuse actualizar el lr a `5e-4` (PR #3934, ya mergeado); las otras dos runs van al graveyard.
* **Frente a LoRA $r=32$:** con `5e-4`, LoHa (0.707, drift 0.256) supera a LoRA $r=32$ en la misma GPU (0.6815, drift 0.2721) y con menos drift.

**Limitaciones:**
* **Qué no prueban:** FedPara es un paper de aprendizaje federado con redes entrenadas desde cero (VGG16, ResNet y LSTM en CIFAR-10/100, CINIC-10 y Shakespeare), no de PEFT sobre modelos preentrenados ni de difusión: su uso como adaptador viene de LyCORIS. Los propios autores avisan de que tantas multiplicaciones (el Hadamard incluido) pueden ser más propensas a gradientes que explotan o se desvanecen y a inestabilidad numérica que el bajo rango clásico (con inicialización He no lo observaron).
* **Supuestos (rango ≠ calidad, y el lr):** con exactamente los mismos parámetros (76,677,120), LoRA $r=64$ da **0.7431** con drift 0.2877 en la L40S: aunque son GPUs distintas, el rango teórico mayor de LoHa (hasta 1024) no se traduce aquí en mejor DINOv2 que LoRA con el mismo presupuesto. Posible explicación de por qué necesita un lr mayor: con Adam, cada paso mueve los parámetros más o menos lo mismo (del orden del lr); en LoRA ese cambio pasa directamente a $\Delta W = BA$, y en LoHa se multiplica elemento a elemento por el otro producto ($A_1 B_1$), que empieza con valores pequeños, así que $\Delta W$ crece más despacio. LoKr, que también multiplica dos factores (Kronecker), necesitó `1e-3`.
* **Coste:** el doble de parámetros que LoRA $r=32$ (76.7M frente a 38.3M), unas **3 veces más lento** (2,281 s frente a 759 s en la L40S) y más memoria (12,104 MiB frente a 10,212 MiB). El paper ya lo reconoce: reconstruir la matriz en cada paso lo hace más lento que el bajo rango clásico. En aprendizaje federado no importa porque domina la comunicación, pero avisan de que en entrenamiento distribuido la ventaja puede desaparecer. La run de `1e-4` es de otra GPU (LoRA $r=32$ da 0.6972 en la L40S y 0.6815 en la 5070 Ti, unas 0.016 de diferencia), pero la mejora de LoHa (0.662 → 0.707) es bastante mayor y `3e-4` ya va en la misma dirección, así que la tendencia no parece ruido de GPU.

**Conexión:** `#peft` `#hadamard` `#high-rank` `#lycoris` `#federated-learning` `#learning-rate` `#flux2-klein` (misma desigualdad de rango que HiRA, pero sin $W_0$; como HiRA y FourierFT, pierde el atajo de LoRA y necesitó un lr mayor; las runs se hicieron con el arreglo del issue #3921 aplicado).

**Preguntas**
1. **Conceptual:** *¿Cómo consigue LoHa rango hasta $r^2$ y en qué se diferencia de HiRA?*
   * **Respuesta:** Multiplicando elemento a elemento dos productos de bajo rango: $\Delta W = (A_1B_1) \odot (A_2B_2)$. Como $\text{rango}(X \odot Y) \le \text{rango}(X) \cdot \text{rango}(Y)$, el rango puede llegar a $r^2$ con solo el doble de parámetros que LoRA. HiRA usa la misma desigualdad, pero multiplica un único producto de bajo rango por los pesos congelados $W_0$, apoyándose en lo que el modelo ya sabe.
2. **Matemática / Diseño:** *¿Por qué LoHa necesita un lr mayor que LoRA para converger en pocos pasos?*
   * **Respuesta:** Con Adam, cada paso mueve los parámetros del orden del lr. En LoRA, ese cambio entra directamente en $\Delta W = BA$; en LoHa, el cambio de un par queda multiplicado elemento a elemento por el otro producto, que empieza con valores pequeños (y uno de los factores empieza a 0), así que $\Delta W$ crece más despacio. Con `1e-4` LoHa daba 0.662; con `5e-4`, 0.707.
3. **Crítica / Trade-offs:** *Con `5e-4` LoHa supera a LoRA $r=32$. ¿Significa eso que su rango alto le da ventaja sobre LoRA?*
   * **Respuesta:** No: LoHa tiene el doble de parámetros que LoRA $r=32$. Con exactamente los mismos parámetros, LoRA $r=64$ da 0.7431 en la L40S, mejor que LoHa (aunque en otra GPU), así que su rango teórico de 1024 no se traduce en mejor DINOv2 con el mismo presupuesto. Además es unas 3 veces más lento al entrenar, porque construye $\Delta W$ entera en cada paso.

**Flashcards**
1. **P:** ¿Cuál es la actualización de LoHa y su cota de rango?
   **R:** $\Delta W = (A_1B_1) \odot (A_2B_2) \cdot \alpha/r$, con $\text{rango}(\Delta W) \le r^2$ (1024 con $r=32$).
2. **P:** ¿Cuántos parámetros tiene LoHa frente a LoRA con el mismo $r$, y qué $r$ necesitaría LoRA para el mismo rango?
   **R:** El doble: $2 \cdot r \cdot (\text{in} + \text{out})$ frente a $r \cdot (\text{in} + \text{out})$. LoRA necesitaría $r=1024$, 16 veces más parámetros que LoHa con $r=32$.
3. **P:** ¿Cómo se inicializa LoHa en `peft` para no modificar el modelo al principio?
   **R:** Tres de las cuatro matrices aleatorias y `hada_w2_b` a ceros: el segundo producto vale 0 y $\Delta W = 0$.
4. **P:** ¿Qué lr elegí para LoHa en `FLUX.2-klein` y por qué?
   **R:** `5e-4`: mejor DINOv2 (0.707 frente a 0.662 con `1e-4`) sin subir el drift; con `1e-3` ya se pasa (drift 0.275, DINOv2 0.697). PR #3934.
